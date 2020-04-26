---
layout: post
comments: true
title: How PyCaffe Implements Parallel Training
---

1. generate the uid
    1. uid = caffe.NCCL.new_uid()

2. For each gpu, create a process solve()
    1. set_device
    2. set_gpu_mode
    3. set_solver_count
    4. set_solver_rank
    5. set_multiprocess as True
    6. nccl = caffe.NCCL(solver, uid)
        1. ncclCommInitRank(&comm_, Caffe::solver_count(), nccl_uid, Caffe::solver_rank()));
    7. solver.add_callback(nccl)
        1. if it is layer_wise_reduce, the callback of nccl does almost nothing
           except the sync
    8. nccl.bcast()
    9. if solver.param.layer_wise_reduce -> solver.net.after_backward(nccl)
    10. solver.step(iters)
    

