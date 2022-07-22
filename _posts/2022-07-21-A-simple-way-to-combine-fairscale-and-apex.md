---
layout: post
comments: true
title: A simple way to combine fairscale and apex
author: Jianfeng Wang
---

Both [deepspeed](https://github.com/microsoft/deepspeed) and 
[fairscale](https://github.com/facebookresearch/fairscale) 
are [pytorch](https://github.com/pytorch/pytorch) 
libraries for large-scale model training by sharding the optimizer
state, gradient, model parameters. 
However, they are not exactly the same.
A key difference is that deepspeed uses fp16, while fairscale uses fp32 for
the model parameters. Normally, deepspeed can save more GPU memory, which
might improve the speed by increasing the batch size.
To reduce the speed gap, this blog presents some background knowledge of
[scaleapex](https://github.com/amsword/scaleapex), which 
provides a simple way such that the fairscale can also use
fp16 to have deepspeed-like training.
One benefit is to customize the training procedures more easily with
fairscale, e.g. different dynamic loss scalers for different losses.

It is worth noting that here fp16 means all parameters in the model are fp16,
which is different from the mixed precision. The mixed precision normally
means that the parameters are fp32, the computations are fp16, and the
activations are fp32. Thus, fp16 can save more memory than mixed precision.

# How fp16 is used in deepspeed
First, let's review how fp16 is used in deepspeed, which is based on
[FP16_optimizer](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer).
The steps are
1. All parameters in `model` are converted to fp16 by `model.half()`. Thus, 
   when we call `loss = model(data)`, all computations are in fp16. All
   intermediate activations are also in fp16.
2. The loss is scaled up by a dynamic loss scaler, i.e. `scale_loss = loss * scaler`. The
   scaler will be increased if there is no NaN for a specific number of
   iterations, but will be decrased if NaN is hit.
3. Run `scale_loss.backward()`, and each fp16 parameter will have fp16 `.grad`.
4. The `optimizer` holds a copy of the model parameters, but in fp32, which
   are called master parameters.
5. All these fp16 `.grad` are copied to the master parameter's `.grad`.
6. Run the optimizer update on those master parameters, and then copy these
   updated master parameters back to the fp16 parameters used in `model`.

# How fairscale works with sharding
For optimizer state sharding, the key implementation is `fairscale.optim.oss.OSS`.
The first two arguments are `params` and `optim`. Inside `OSS`, 
1. Each parameter group will be divided into `N` disjoint groups, where `N`
   is the number of GPU workers or the world size. Thus, it supports multiple
   parameter groups.
2. Construct a new parameter group based on the rank ID. Let's say it is `curr_partition_parameters`.
3. Call `curr_optim = optim(curr_partition_parameters)` such that we have an optimizer which
   only sees a portion of the parameters. `optim` is the second argument of
   `OSS`.
4. When we call `optimizer.step()`, 1) it will call the `curr_optim` to update the
   partitioned parameters, and 2) broadcast the parameters such that all GPUs
   have the same updated parameters.

If we convert all model to fp16, the optimizer of `curr_optim` will also be fp16, which has no
fp32 master parameters. Thus, the key is to hack the `optim` parameter such that the 
master parameter can be used for real parameter updates.

# Hack it
As described above, we can provide a special `optim` such that the created
optimizer inside OSS can use fp32 master parameters for fp16 update. 
Here is the key code path with the tool of [scaleapex](https://github.com/amsword/scaleapex), 
and a full example can be found [here](https://github.com/amsword/scaleapex/blob/main/example/example.py)

```python
from scaleapex import optim_constructor
from fairscale.optim.oss import OSS
from functools import partial

extra_param = {
    'lr': 1.e-7,
    'weight_decay': 0.01,
}
optimizer = OSS(
        parameters,
        optim=partial(optim_constructor, AdamW),
        **extra_param,
        )
```






