---
layout: post
comments: true
title: Is the data loader fast enough in Pytorch?
author: Jianfeng Wang
---


When we train the deep learning model in Pytorch, we may hit the
issue of slow speed, especially in the multi-node distributed training. One
critical issue is to make sure that the data loader is fast enough and to
improve it if it is slow. This blog
will present some tips on how to verify this.

## Baseline
A general training pipeline can be as follows: load `data` from the data loader
and then update the model parameter based on the `data`.
```python
for i, data in enumerate(data_loader):
    network_forward_backward_update(data)
```
To check whether the data loader is fast enough, we need to calculate the time
cost over the data loading. The baseline approach is
```python
start = time.time()
for i, data in enumerate(data_loader):
    time_data = time.time() - start
    network_forward_backward_update(data)
    print(time_data)
    start = time.time()
```
The time cost is printed out every iteration. To reduce the log volume, we
typically print it every several iterations, e.g. 100. That is
```python
start = time.time()
for i, data in enumerate(data_loader):
    time_data = time.time() - start
    network_forward_backward_update(data)
    if (i % log_step) == 0:
        print(time_data)
    start = time.time()
```

## Summarization since last print
However, this logging cannot capture the case if the data loader is slow
between two consecutive prints. Thus, the printed time cost
should be some summarization over all iterations since last print rather than
for only the current iteration. We have the following update.

```python
start = time.time()
all_time_data = []
for i, data in enumerate(data_loader):
    all_time_data.append(time.time() - start)
    network_forward_backward_update(data)
    if (i % log_step) == 0:
        print(sum(all_time_data) / len(all_time_data))
        all_time_data = []
    start = time.time()
```

## Relative time cost
The absolute time cost may not make too much sense without a reference, so we
can print the total time cost. If the relative time cost is low, we can
conclude it is fast enough
```python
start = time.time()
all_time_data = []
all_time_iter = []
for i, data in enumerate(data_loader):
    all_time_data.append(time.time() - start)
    network_forward_backward_update(data)
    all_time_iter.append(time.time() - start)
    if (i % log_step) == 0:
        print('{}/{}'.format(
            sum(all_time_data) / len(all_time_data),
            sum(all_time_iter) / len(all_time_iter),
        ))
        all_time_data = []
        all_time_iter = []
    start = time.time()
```

## Log from the master worker
Until now, we use `print` to log the time cost. In multi-GPU training, e.g. 256
GPUs, there will be 256 log entries each time when we print the time cost. This could
be annoying. However, if we only print
the log on the master worker, we may not capture the speed issue from
non-master workers. To address the issue, we can only print the log if the
ratio of the data time cost is not small enough. However, we have to pre-define a
threshold, which might be sensitive to the application. Anyway, let's give the
full implementation  here.

```python
start = time.time()
all_time_data = []
all_time_iter = []
th = 0.01
for i, data in enumerate(data_loader):
    all_time_data.append(time.time() - start)
    network_forward_backward_update(data)
    all_time_iter.append(time.time() - start)
    if (i % log_step) == 0:
        avg_data = sum(all_time_data) / len(all_time_data)
        avg_total = sum(all_time_iter) / len(all_time_iter)
        if avg_data > th * avg_total or is_master_worker():
            print('{}/{}'.format(
                sum(all_time_data) / len(all_time_data),
                sum(all_time_iter) / len(all_time_iter),
            ))
        all_time_data = []
        all_time_iter = []
    start = time.time()
```

Another tip is to print out the medium as well as the average. Sometimes, we
may hit the situation where 1) the mean value is high and 2) the medium value is small.
This normally means that some samples are offensive and need lots of time. One
example is that most of the images are small, but some few images are super
bit, which needs lots of time on I/O and preprocessing in the data loader.

## Conclusion
The important parts are that 1) the printed or verified time cost should be the
summarization since the last print and 2) time cost on non-master worker also
needs to check as most of the time only the master worker's performance is
examined.

