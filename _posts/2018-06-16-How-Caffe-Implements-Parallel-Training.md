---
layout: post
comments: true
title: How Caffe Implements Parallel Training
published: false
---

```cpp
caffe::NCCL<float> nccl(solver);
nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
```

1. caffe::NCCL<float> nccl(solver); 
	1. GPUParams<Dtype>(solver, getDevice())
		1. Params<Dtype>(root_solver)
			1. Get the size of the learnable parameters
			2. No space allocation on two pointers: data_ and diff_
		2. Create space on GPU for data_ and diff_, based on the size calculated in Params()
			1. The space is allocated on the input device id
			2. The parameter space is continous
		3. Copy the network parameters from cpu to the data_
	2. Let all the pointers in the learnable parameters point the space in data_ and diff_, 
        1. Since the data are continous, we can use one MPI or NCCL command to
           transfer all the data. We can also use multiple commands to transfer
           multiple times, each of which transfers only part of the data.
	3. Set multiprocess=True
	4. If layer_wise_reduce -> set nonblocking attribute on the cuda
2. nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
	1. barrier(static_cast<int>(gpus.size()))
        1. It is used to sync all the workers, including the master worker
	2. nccls(gpus.size());
        1. Each nccl has the pointer of solver, and almost everything.
	3. Create len(gpus) - 1 workers. 
		1. For each worker
			1. Pass the rank to the singlton before starting the thread
			2. The internal_thread
				1. feeds the rank to the worker. 
				2. Create a solver s
				3. Create a nccl(s)
				4. feed the barrier to nccl
                5. add nccl to the callback of current solver and the network
                   if layer-wise-reduce
                    1. s->add_callback(&nccl)
                    2. if (s->param().layer_wise_reduce()) { s->net()->add_after_backward(&nccl); }
				6. Register the nccl to nccls, which will be used by master
                   thread
                    1. (*nccls_)[Caffe::solver_rank()] = &nccl;
				7. Barrier.wait()
				8. Barrier.wait()
                9. nccl.Broadcast();
                    1. ncclBcast(data_, static_cast<int>(size_), nccl::dataType<Dtype>::type, 0, comm_, cudaStreamDefault);
                        1. Broadcast the data from rank 0 to others. 
                        2. These data are the network parameters
                10. s->Step(param.max_iter() - s->iter());
                11. barrier_->wait();
    4. add current nccl to the callback of the solver and the network
        1. solver_->add_callback(this);
        2. if (solver_->param().layer_wise_reduce()) {
            solver_->net()->add_after_backward(this);
            }
	5. Barrier.wait()
		1. Each worker will initialize a solver. Waiting the solver to finish the initailization since we will get the device id from the solver in the nccl initialization
	6. InitSingleProcess()
		1. Get communicator for all nccls (including rank 0)
		2. Register the communicator to each nccl object in nccls
		3. It needs to access the nccl in nccls, so nccls should be initialized. This is the reason why we need a barrier before this function
	7. Barrier.wait()
		1. This is more used to unblock the other worker threads, since they should not move forward unless the nccl communicator is initialized. 
    8. Broadcast();
    9. solver_->Solve();
    10. Interrupt each worker


Callback. 
1. The parameter of solver_->add_callback is Solver::CallBack.
    1. NCCL object is the parameter
    2. class NCCL: public Solver<Dtype>::Callback;
    3. Solver::CallBack. 
        1. NCCL needs to implement on_start and on_gradient_ready function
            1. on_start() is empty in NCCL's implementation.
            2. on_gradients_ready():
                1. if it is layer_wise_reduce, do nothing except sync
                2. if it is not layer_wise_reduce:
                    1. ncclAllReduce(diff_, diff_, static_cast<int>(size_), nccl::dataType<Dtype>::type, ncclSum, comm_, cudaStreamDefault)
                    2. caffe_gpu_scal(static_cast<int>(size_), (Dtype) 1.0 / Caffe::solver_count(), diff_);
        2. on_start is called before forward for each iteration
        3. on_gradient_ready is called after backward and before update the
           parameters
            1. in ApplyUpdate() to update the parameters
                1. regularize the parameters.
                2. computer the update value also based on the momentum
                   parameters
                3. update all the parameters
    4. Note the parameter is not broadcasted every iteration, since each worker
       will have the same diff and they will apply the parameter update
       independently with the same data. So, there is no need to do that. For
       BN layer, each worker does the BN logic independently and the running
       mean and running variance is also updated independently. When we save
       teh model, we only use the running mean/varaiance from rank 0.
1. The parameter of net->add_after_backward is Net::CallBack. 
    1. NCCL object is the input parameter;
    2. class NCCL: public Net<Dtype>::Callback;
    3. Net::CallBack requires the derived class to implement Run(layer_id)
    4. It is called after the backward on each layer is done. only if
       layer_wise_reduce
    5. How NCCL implements that
        1. Each layer may have multiple blobs. Thus, first to count the number
           of elements we need to reduce
        2. Since we are using continous space, we can just call ncclAllReduce
           once and then do the scaling 
            1. ncclAllReduce(blobs[0]->mutable_gpu_diff(),
                                     blobs[0]->mutable_gpu_diff(),
                                     size,
                                     nccl::dataType<Dtype>::type,
                                     ncclSum, comm_, stream_)
            2. caffe_gpu_scal(size, (Dtype) 1.0 / Caffe::solver_count(),
                           blobs[0]->mutable_gpu_diff(), stream_);
		
