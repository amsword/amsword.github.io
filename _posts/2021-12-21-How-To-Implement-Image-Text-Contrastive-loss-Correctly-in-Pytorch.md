---
layout: post
comments: true
title: How to implement the image-text contrastive loss correctly in Pytorch
author: Jianfeng Wang
---

The image-text contrastive (ITC) loss is a simple yet effective loss to align the paired
image-text representations, and is successfully applied in OpenAI's
[CLIP](https://arxiv.org/abs/2103.00020) and
Google's [ALIGN](https://arxiv.org/abs/2102.05918).
The network consists of one image encoder and one text encoder, through which
each image or text can be represented as a fixed vector.
The key idea of
ITC is that the representations of the matched images and texts should be as close
as possible while those of mismatched images and texts be as far as possible.
The model can be well applied to the retrieval task, classification task, and
others replying on an image encoder, e.g. object detection.

Recently, I find it is not that easy to implement the loss correctly in
Pytorch, especially in the distributed environment. 
## Baseline
Let's firstly take a
look at a baseline implementation. In each iteration, we generate the image
representation `image_feat` and the text representation `text_feat`. Both of
the representations contain `N` rows and each row is a `D`-dimensional vector.
Then, we can have the following implementation.

```python
def image_text_contrastive_loss_baseline(image_feat, text_feat, temperature):
    N = image_feat.shape[0]
    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature
    gt = torch.arange(N, device=logits.device)
    loss1 = torch.nn.functional.cross_entropy(logits, gt)
    loss2 = torch.nn.functional.cross_entropy(logits.t(), gt)
    return (loss1 + loss2) / 2
```
## L2 normalization
One important thing is that the L2 normalization should be applied to the
features before applying the loss function. This is essential regarding the
accuracy, e.g. retrieval performance. To make it complete, we have the
following implementation.

```python
def image_text_contrastive_loss_with_l2(image_feat, text_feat, temperature):
    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)
    N = image_feat.shape[0]
    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature
    gt = torch.arange(N, device=logits.device)
    loss1 = torch.nn.functional.cross_entropy(logits, gt)
    loss2 = torch.nn.functional.cross_entropy(logits.t(), gt)
    return (loss1 + loss2) / 2
```
## Duplicate images or texts
What if there are two identical images or texts? For example,
each image in the [COCO](https://cocodataset.org/#home) dataset contains 5 text descriptions,
and thus there is a chance that two different rows in `image_feat` correspond to the same image.
For the concept of being identical, we only consider the case where two images
are with the same image index in the dataset, and where the text description strings are the same. We do not consider the case if the two images or
the two texts are not the same but super similar because it is hard to robustly tell
whether these two are similar enough.

In the baseline implementation, we assume only one positive for each image or
the text. Considering duplicates, we need to handle multiple positives. Here we
just use the multi-hot cross-entropy loss. 
1. For each image, we can use the
image index in the dataset as the identifier, and the two images are identical if and
only if the indices are the same. It is noted that in the dataset of
image-text pairs, we have the concept of the image index and the concept of the image-text index.
2. For each text description, we can generate a hash value for each string, and the two texts are considered to be the same if the hash values are
identical. As the hash function might not be perfect, there is a chance that
the two text descriptions are different but the hash codes are the same. As this conflict is rare, we will simply assume we have a perfect hash function.
Practically, we can generate the hash code from the data loader in the PyTorch
training pipeline. One hash function we can use is the built-in `hash(str)` in
python.

After we have the identifiers, we have the following implementations

```python
def image_text_contrastive_loss_with_l2_id(image_feat, text_feat, temperature, image_id, text_id):
    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)
    N = image_feat.shape[0]
    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature

    gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
    gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
    gt = torch.logical_or(gt_image, gt_text)

    loss1 = -torch.sum(gt * torch.nn.functional.log_softmax(logits, dim=1)) / gt.sum()
    loss2 = -torch.sum(gt.t() * torch.nn.functional.log_softmax(logits.t(), dim=1)) / gt.sum()
    return (loss1 + loss2) / 2
```
## Distributed training
### Gather representations
Almost always, we use multiple GPUs to run the contrastive loss as it
is normally for large-scale training. In distributed training, each GPU
processes only a portion of the data and then calculates the gradient itself
before averaging the gradient across all participated GPUs. As the loss is
based on the batch size, we need to gather the features before applying the
loss. One function we need is `torch.distributed.all_gather()` to gather all
features and the identifiers. The first parameter is a list of tensors that
are going to be populated, and the second is the tensor we will gather. Here,
we assume the tensor shape is the same across all GPUs and we have the following
wrapper function.
```python
def all_gather(x):
    all_x = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_x, x)
    x = torch.cat(all_x, dim=0)
    return x
```
If the shape of the input is $$N\times D$$ and we have $M$ GPUs, the returned
tensor's shape is $$(NM)\times D$$. Thus, each GPU can have all representations
by `image_feat=all_gather(image_feat)` and `text_feat=all_gather(text_feat)`.

### Gradient backpropagation
However, this function does not send back the
gradient. To verify this, we can have the following test:
```python
import os, torch
torch.cuda.set_device(int(os.environ.get('OMPI_COMM_WORLD_RANK', '0')))
torch.distributed.init_process_group(
    backend='nccl', init_method='tcp://localhost:12345',
    rank=int(os.environ.get('OMPI_COMM_WORLD_RANK', '0')),
    world_size=int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')),
)
x = torch.zeros((5, 5), device='cuda', requires_grad=True)
print(x.requires_grad) # True
x = all_gather(x)
print(x.requires_grad) # False
```
Let's say the script is `script.py`. We can run it by `python script.py` or
`mpirun -n 2 python script.py` if at least 2 GPU devices are available to use.

To address the issue, we can re-use the current `x` to replace the gathered current `x`. 
The reason is that the current `x` can propagate back the gradient. Overall, we
can have the following
```python
def all_gather_grad(x):
    all_x = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_x, x)
    all_x[torch.distributed.get_rank()] = x # essential to propagate gradient on x
    x = torch.cat(all_x, dim=0)
    return x
```
That is, before `image_text_contrastive_loss_with_l2_id()` or at the very
beginning of this function, we should call `all_gather_grad` to gather the full representations as follows
```python
def image_text_contrastive_loss_with_l2_id_gather(image_feat, text_feat, temperature, image_id, text_id):
    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)

    # add the following 4 lines
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    image_id = all_gather_grad(image_id)
    text_id = all_gather_grad(text_id)

    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature

    gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
    gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
    gt = torch.logical_or(gt_image, gt_text)

    loss1 = -torch.sum(gt * torch.nn.functional.log_softmax(logits, dim=1)) / gt.sum()
    loss2 = -torch.sum(gt.t() * torch.nn.functional.log_softmax(logits.t(), dim=1)) / gt.sum()
    return (loss1 + loss2) / 2
```
However, this implementation is still NOT correct.

### Scale properly the loss
After we use `all_gather_grad()` to have the full representation, we need to
keep in mind that the gradient on the other GPUs' representation will NOT be
propagated back but only that on the current GPU's representation be. Let $$M$$
be the number of GPUs and $$x_i$$ be the representations (both the image and the text) on the $$i$$-th GPU.
Then, we can write the loss in a general form as $$f(x_1(\theta), \cdots, x_m(\theta))$$, where $$\theta$$ denotes 
all the learnable parameters. Then, the gradient should be

$$
\frac{\partial f}{\partial \theta} = \sum_{m} \frac{\partial f}{\partial x_m}
\frac{\partial x_m}{\partial \theta}
$$

As the gradient on the other GPUs' representation can not be propagated back,
each GPU actually accidentally calculates $$\frac{\partial f}{\partial x_m} \frac{\partial x_m}{\partial\theta}$$.
During gradient synchronization (e.g. `torch.nn.parallel.DistributedDataParallel`), the average reduction is performed, and thus we need to
scale up the loss value by $$M$$ to finalize the correct gradient. That is

```python
def image_text_contrastive_loss(image_feat, text_feat, temperature, image_id, text_id):
    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)

    # add the following 4 lines
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)
    image_id = all_gather_grad(image_id)
    text_id = all_gather_grad(text_id)

    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature

    gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
    gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
    gt = torch.logical_or(gt_image, gt_text)

    loss1 = -torch.sum(gt * torch.nn.functional.log_softmax(logits, dim=1)) / gt.sum()
    loss2 = -torch.sum(gt.t() * torch.nn.functional.log_softmax(logits.t(), dim=1)) / gt.sum()
    return (loss1 + loss2) / 2 * torch.distributed.get_world_size() # scale it up by the number of GPUs
```

## Conclusion
At the first glance, the image-text contrastive loss should be easy to
implement. However, there are lots of details to make it right, and the most
important could be the last one: scale it up by the number of GPUs. To make it
easy to use, here is the [code](https://github.com/amsword/itc) and you can
apply it

1. Install it by
```bash
pip install https://github.com/amsword/image_text_contrastive
```
2. Use it
```python
from image_text_contrastive import image_text_contrastive_loss as itc
itc(image_feat, text_feat, temperature, image_id, text_id)
```
