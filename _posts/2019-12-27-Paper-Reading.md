---
layout: post
comments: true
title: Paper Reading
---

- To read
    - Conditional image generation with pixelcnn decoders
    - Population Based Training of Neural Networks
    - Algorithms for Hyper-Parameter Optimization
    - random search for hyper-parameter optimization
    - Snapshot Distillation: Teacher-Student Optimization in One Generation
    - Learning to compose domain-specifici transformations for data
      augmentation.
    - Object detection via a multiregion & semantic segmentation-aware cnn model
    - Web-Scale Responsive Visual Search at Bing
    - Wasserstein GAN
    - Image-to-image translation with conditional adversarial networks
    - https://venturebeat.com/2020/02/11/researchers-develop-technique-to-increase-sample-efficiency-in-reinforcement-learning/
    - Generative adversarial nets
    - Conditional generative adversarial nets
    - U-net: Convolutional networks for biomedical image segmentation
    - Progressive pose attention transfer for person image generation
    - Differentiable learningto-normalize via switchable normalization
    - Disentangled Person Image Generation
    - Web-Scale Responsive Visual Search at Bing
    - When Unsupervised Domain Adaptation Meets Tensor Representations
    - Learning Transferable Features with Deep Adaptation Networks
    - Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks
    - unsupervised domain adaptation by backpropagation
    - Domain-Adversarial Training of Neural Networks
    - Few-shot Object Detection via Feature Reweighting
    - Meta-RCNN: Meta Learning for Few-Shot Object Detection
    - https://towardsdatascience.com/few-shot-learning-in-cvpr19-6c6892fc8c5
    - Robust scene text recognition with automatic rectification
    - Convolutional Sequence to Sequence Learning
    - DARTS: Differentiable Architecture Search
        - https://github.com/quark0/darts
    - symmetry-constrained rectification network for scene text recognition
    - XNAS: Neural Architecture Search with Expert Advice
    - Progressive Neural Architecture Search
    - Non-Local Neural Network
    - CornerNet: Detecting Objects as Paired Keypoints
    - GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
    - Bottom-up Object Detection by Grouping Extreme and Center Points
    - Side-Aware Boundary Localization for More Precise Object Detection
    - Feature Selective Anchor-Free Module for Single-Shot Object Detection
    - Efficient Object Detection in Large Images using Deep Reinforcement Learning
    - Dually Supervised Feature Pyramid for Object Detection and Segmentation
    - SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
    - IoU-uniform R-CNN: Breaking Through the Limitations of RPN
    - AugFPN: Improving Multi-scale Feature Learning for Object Detection
    - RDSNet: A New Deep Architecture for Reciprocal Object Detection and Instance Segmentation
    - Learning from Noisy Anchors for One-stage Object Detection
    - RefineNet: multi-path refinement networks for dense prediction
    - FCOS: Fully Convolutional One-Stage Object Detection
        - github: https://github.com/tianzhi0549/FCOS
    - NAS-FCOS: Fast Neural Architecture Search for Object Detection
        - https://github.com/Lausannen/NAS-FCOS

# Data
- Connecting Vision and Language with Localized Narratives
    - provide annotations on coco (all, 100k+) and part (504k) of open image dataset
    - an annotator is required to 1) describe the image content by voice 2)
      hover the mouse on the image region, synchronized with the voice, 3) write
      the transcription immediately after annotating one image.
      - to align the transcript to the voice, (then to the mouse trace/image
        region)
        - automatically generate a transcription of the spoken caption.
        - align the manually written transcription with the auto transcription
          which has the timestamp information
          - the alignment keeps the order.
          - the alignment is based on the edit distance between words.
    - 40.6 seconds for speaking; 110.2 for writing the transcript.
    - manually tried the annotation process, and it is not that natural. maybe
      combine with the click event or draw event (click-drag-clik)? Otherwise,
      there might be some redundant trace between objects, which might not be
      useful and not be easy to remove.


# Network Component
- Making Convolutional Networks Shift-Invariant Again
    - ICML 19
    - The motivation is that the downsampling layer (pool and conv with
      stride) could make output variant with input shift. The solution is
      to apply a smoothing filter before sampling.
        - smoothing filter or anti-aliasing filter is equivalent with a
          conv layer with the following parameters
            - the group size is the same as the input size. That is, each
              input feature map is processed independently
            - the output feature number is the same as the input feature
              number. That is, each input feature map generates one output
              feature.
            - the kernel is a pre-defined parameter, which behaves like a
              low-pass filter, e.g. outer product of [1, 2, 1] and [1, 2, 1]^T
                - note, since it is a pre-defined paraameter, one natural
                  extension is to make it learnable! But i did not see the
                  experiment with this.
            - normalization is used.
        - max pooling with stride 2
            - this layer is converted to
                - 1) max pooling with stride 1,
                - 2) apply a smoothing filter or anti-aliasing filter
                - 3) sub-sampling.
        - conv with stride 2 and relu
            - this layer is converted to
                - conv with stride 1
                - relu
                - apply a smoothing filter
                - sub-sampling
            - it combines with relu, and it is not like the following
                - conv with stride 1
                - apply a smoothing filter
                - sub-sampling
                - relu
            - the reason should be that conv is a linear operation;
              smoothing is also a linear operation. Thus it would be
              equivalent to have one learnable conv layer as long as the
              kernel size of smoothing filter is smaller than/equal to the
              conv kernel size.
            - the computational time cost of such converted conv operation
              will be doubled at least since conv with stride 2 becomes
              with conv with stride 1 first. It would be hard to optimize
              since the non-linear layer is applied afterwards.
        - average pooling with stride 2
            - this layer is converted to
                - smoothing filter
                - sub-sampling with stride 2
    - Experiment
        - in imagenet, it has 0.5-1 point gain.
        - the inference time should be increased, but there is no report in
          the paper. it only applies on the down-sampling layer, and thus
          maybe not too much, 10% maybe?
        - the smoothing kernel can all become learnable since it is one of
          special kinds of conv layer with group and fixed kernel. no
          report of such experiment in the paper. Learning time could be
          longer, but inference time should be the same.
        - not sure how it performs in detection and other tasks.

# Image Classification
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    - ICML 2019
    - contribution
        - scale the network width (channels), depth, and input image size at the same time
    - others
        - apply an existing work (same author) to search a network with fewer target flops and 
          use flops as the cost measurement rather than hardware cost
            - MnasNet: Platform-aware neural architecture search for mobile
- ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    - ECCV 18

- Bayesian Optimization
    - https://github.com/fmfn/BayesianOptimization
        - github star is 3.8k
        - good to use; checkout the advanced tutorial, with
          suggest-evaluate-register
    - https://ax.dev/
        - github star is 1k
        - from facebook
        - good to use. checkout the service api example
    - https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
- network component
    - In-Place Activated BatchNorm for Memory-Optimized Training of DNNs
        - an efficient BN implementation.
        - worth giving it a try
        - 2018, CVPR

- network architecture search
    - MnasNet: Platform-aware neural architecture search for mobile
    - Neural Architecture Search: A Survey
        - high-level introduction of the approaches from 3 aspects: the search space,
          search strategy, and performance evaluation
        - pretty-good survey

- Few-short learning
    - Generalizing from a Few Examples: A Survey on Few-Shot Learning
        - archiv, 5/13/19. worth reading next time

# Self-Supervised Learning
- Improved Baselines with Momentum Contrastive Learning
    - arxiv 3/9/2020
    - incorporate the tricks shown in A Simple Framework for Contrastive Learning of Visual Representations
      to the momentum constrastive learning, i.e. add extra MLP layer for
      pre-training, more data augmentation, more iterations.
        - the accuracy is improved from 60.6 to 71.1.
- S4L: Self-supervised semi-supervised learning
    - problem: target supervised classification dataset (1% of the imagenet
      dataset or 10%) with the unlabeled image datasets. The solution is to
      train the model on the joint of the two datasets with the loss from
      unsupervised learning. One is to use the rotation loss, i.e. randomly
      rotate the input image with 0/90/180/270 degres and predict the degree
      with the cross entropy loss. The other is some exemplar loss, which is
      similar with the contrastive loss. The experiments show that with 1% of
      the labelled data,
        - the approach of training supervised model only on the labelled data
          gives 48.43% top-5 accuracy
        - the approach of psuedo labeling (train a model on labeled, propagate
          the labels on the unlabeled, re-train the model on the full) gives
          51.56.
        - self supervised + linear layer gives at most 25.98 accuracy
        - self supervised + fine-tune gives at most 45.11, which is even lower
          than the 48.43, which is from the supervised learning only. 
        - the joint training gives 53.37 with the rotation loss.
- Learning Representations by Maximizing Mutual Information Across Views
    - NIPs 2019, MSR Montreal. 3 authors, the first two are also the authors of
      the paper which this paper is based on
    - the contribution is to extend the paper of LEARNING DEEP REPRESENTATIONS 
      BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION by introducing
        - multi-view of the images to construct the loss
            - multi-view comes from multiple instantiation of the data augmentation
            - the baseline is to use one instantiation
        - multiple feature maps from different spatical sizes are used rather
          than 1.
        - extend the representation of a notion of mixed representation.
    - experiments
        - on cifar10, the baseline is 75.21; while this paper achieves 89.5.
        - on cifar100, the baseline is 49.74, while this paper achieves 68.1
- LEARNING DEEP REPRESENTATIONS BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION
    - iclr 2019, MSR montreal, Yoshua
    - the contribution is to incorporate local features in the representation
      learning. Before, each image correponds to one vector, e.g. R^1024. Now,
      it correponds to multiple vectors, e.g. R^{7x7x104} from the last feature
      map before global average pooling.
      - the objective is to maximize the esitmated mutural information between
        the input and the output, as claimed. But actually, it applies the
        maximization between the local features (the features in 7x7 feature map)
        and the global features (after average pooling).
    - no experiments on imagenet, but on cifar and shrinked version of
      imagenet. The accuracy is comparable with contrastive predictive coding.
- DATA-EFFICIENT IMAGE RECOGNITION WITH CONTRASTIVE PREDICTIVE CODING
    - iclr 2020 submisssion but rejected. appear in arxiv 12/2019
        - the review comment of why it is rejected is lack of novelty.
        - the experiment results are interesting to learn.
        - the paper is not that well written unless you are familiar with the
          paper it is based on.
    - novelty
        - based on the paper of Representation Learning with Contrastive Predictive Coding,
          this paper introduces more tricks to boost the accuracy and apply it
          on few-shot learning or low-labeled tasks.
          - tricks
            - more powerful network, named resnet161. original implementation
              uses resnet101. --> 5 point gain
            - originally, each patch's size is 64x64. in this paper, it is
              increased. --> 2 point gain. the paper does not tell the
              increased resolution
            - originally, BN is not used, and the reason might be that the
              pretraining works on the patch, but the testing works on the full
              image. The BN captures the input statistics and is thus not
              appropriate in such settings where train and inference are
              different. in this work, it uses layer normalizatoin. -> 2 point
              gain. This paper does not tell if other normalization (not related with 
              batch size) has similar gains.
            - originally, the image patches in the upper side predict the
              regions in the below. This paper extends it to other directions:
              lower predicts upper; left predicts right; right predicts left.
              By using two directions, -> 2 point gain. By using 4 directions,
              2.5 point gains.
            - apply more augmentation on each patch, e.g. color dropping -> 3
              point gain. including other augmentations -> 4.5 point gains.
    - experiments (this may also be the novelty)
        - on the task of imagenet pretraining + imagenet classificatoin with
          one linear layer, it improves from 48.7 to 71.5
        - on the few shot classification task (using 1% of the imagenet data),
          the tricks which works on using 100% imagenet data is not necessary
          to work here.
          - by using 1%, the fully supervised gives 44.1, which is improved to
            77.1 by the pre-training+finetuning. The reason may be that the
            pre-training could see more data.
        - the results on voc detection are 76.6% given resnet161, while the
          supervised counterpart is 74.7.
- Representation Learning with Contrastive Predictive Coding
    - deepmind, arxiv 1/2019
    - contribution
        - a framework of contrastive predictive coding for feature learning
            - assume the input signal is x_t
            - encode x_t as z_t=g_enc(x_t)
            - calculate context representatoin as c_t = g_ar(z_<=t). That is,
              calculate a representation based on the signal from the time
              earlier or equal to t.
            - use c_t to predict x_{t+k} by contrastive loss. That is, maximize
              f(x_{t+k}, c_t) = z_{t+k}^T W_k c_t compared with the sum of
              f(x_{j}, c_t), for all j.
        - apply the framework to speech, images, text, rl
            - in imagenet, the accuracy is 48.7.
                - each image is split into 7x7 overlapped regions
                - each region is encoded by g_enc to get z_{u, v}
                    - The encoder of g_enc is based on resnet v2 101.
                - the autoregressive model accepts the regions from top to 
                  predict the regions in the below.
                    - that is, c_{u, v} based on z_{x, v}, (x <= u)
                    - the g_ar is based on PixelCNN-style autoregressive model.
                - in sec 2.1 of DATA-EFFICIENT IMAGE RECOGNITION WITH CONTRASTIVE
                  PREDICTIVE CODING gives a clearer explanation of the details on vision
- A Simple Framework for Contrastive Learning of Visual Representations
    - arxiv 2/2020
    - Hinton
    - contribution
        - add non-linear layer after the representation before applying the
          contrastive loss
        - more iterations
        - larger batch size
    - others
        - image classification problem
            - pre-training + fine-tining last layer
        - imagenet2012
- Self-labelling via simultaneous clustering and representation learning.
    - iclr 2020
    - vgg, oxford
    - key idea
        - alternative update the network parameters and the pseudo labels
            - given the psuedo label, update the network parameters by SGD
            - given the network output, minimize the cross entropy loss
              under the requirement that the labels be balanced distributed
              to optimize the psuedo labels. In this step, the label is
              discrete, and thus needs to be relaxed.

# Learning from auxiliary supervised labels
- Caption Datat --> Detection
    - Cap2Det: Learning to Amplify Weak Caption Supervision for Object Detection
        - Problem
            - use the caption dataset to infer the image-level labels, which
              are used as input to the weakly-supervised object detection
              algorithm for detection
        - algorithm
            - Novelty: a method of how to infer the image-level labels from the caption
              text
                - each word is embeded by GloVe algorithm, resulting 300D
                  feature
                - the embedding is project to a 400D space
                - max pooling over multiple words in the sentence
                - softmax for the target category domains (e.g. 80 for coco). A
                  linear layer should be applied before the softmax (not
                  mentioned in the paper).
            - use OICR as the weakly supervised learning algorithm after
              infering the image labels. The algorithm is slightly modified
                - in OICR or WSDDN, two softmax operations are applied across
                  proposals and classes, but here one of the operations is
                  replaced as sigmoid. The difference is not studied in the
                  experiment. The re-weighting in OICR is removed here, and the
                  accuracy is not changed substantionaly as claimed but without
                  experiment result support.
        - experiment
            - coco-caption dataset and flicker caption datasets
                - assume these two datasets are carefully annotated with
                  captions. The cost might be more than the image-level labels.
                  In experiments, no studies are shown with noisy caption
                  datasets, e.g. CC, or randomly crawled internet data.
            - for the method of extracting the image-level labels from the
              caption, the proposed method improves from 39.9 (exact term
              matching), 41.7 (learned GLoVe) to 43.1 on voc test by leveraging coco. If using
              Flicker, it is 31.0 (exact term matching) to 33.6 (the results of
              learned GLoVE is not disclosed).

- Semi-supervised training
    - Billion-scale semi-supervised learning for image classification
        - arxiv only 5/2019
        - key idea
            - train a classifier on the supervised data
            - propagate the labels on the unlabeld data
            - pre-train the student network on the propagated dataset
            - fine-tune on the supervised data
        - Comopared with Data Distillation: Towards Omni-Supervised Learning
            - the ref paper studied the approach on detection and keypoint
              detection, while this paper studies the approach on image
              classification.
            - the ref paper re-train on the mixed two dataset, while this paper
              seperates them into two stages: pre-train on the propagated data
              and then fine-tune on the labeled data
    - Data Distillation: Towards Omni-Supervised Learning
        - cvpr18
        - key idea
            - leverage the unlabeled data to help the supervised training
            - method
                - train a teacher network on the supervised data
                - propagate teh labels on the unlabeled data by multi transform
                  inference
                    - multi transform inference means to do inference on multi form
                      of the input, e.g. by multi scaling
                    - the label is hard label and teh threshold is set so that the
                      average number of labels on unlabeled image is similar with that
                      in labeled dataset.
                - re-train the model on the combined data from labeled dataset and
                  the auto-labeled dataset.
        - experiments are on keypoint detection and object detection.
            - The gain for keypoint detection is large
            - teh gain on detection is around 1 point.

# Data Augmentation
- Fast AutoAugment
    - NIPs 2019
    - The key idea is to train a model without data augmentation first; and
      then evaluate the model on the pre-reserved data with different data
      augmentation. That is, the performance of different poicies is not
      evaluated by re-training, but by evaluation
    - on Imagenet, the accuary with R50 is 22.4. Baseline is 23.7; AutoAgument
      gives 22.4 also.
- Learning Data Augmentation Strategies for Object Detection
    - apply the auto augmentation idea from classification to detection.
    - the gain is 2.3 on R50+RetinaNet
    - rotation is perfererred in the augmentation searching
- Population based augmentation: Efficient learning of augmentation policy schedules
    - icml 2019

# Optimization
- Practical Bayesian Optimization of Machine Learning Algorithms
    - not a good tutorial.

# Teacher-Student
- Snapshot Distillation: Teacher-Student Optimization in One Generation
    - cvpr 19
- Improving Fast Segmentation With Teacher-student Learning
    - BMVC 2018
    - scematic segmentation problem, not instance segmentation, but it seems
      like it can also be applied to instance segmentation.
    - the loss
        - traditional loss
        - alignment based on the probability map, similar like the traditional
          soft-label alignment loss
        - alignment based on consistency loss
            - for student and teacher network, compute the first order
              information, i.e. for each spatial location, calculate the mean
              of the difference between its response and the neighbor's
              response. If it is smooth, the information should be 0. If it is
              near the edge, the value should be large.
            - then add an alignment loss on the first order information
            - equivalent to have a higher weight on edge area
        - add extra training data for training
    - experiment
        - no teacher -> 40.9
        - add soft-label alignment loss -> 42.3
            - this is where the gain comes from most.
        - add that consistency loss -> 42.8
        - add extra training image -> 43.8
- Learning Efficient Detector with Semi-supervised Adaptive Distillation
    - arxiv 1/2019. not find if it is in peer-reviewed conf/journal
- Mask Guided Knowledge Distillation for Single Shot Detector
    - icme 2019
    - previous methods focus on the region features to be aligned
      between the teacher's and the student's. This paper combines the
      global feature alignment and this local feature assignments.
    - Based on SSD
    - with global feature alignment, it is 54.6%. The baseline is not
      reported by not aligning the feature. with the global + local
      feature alignment, it is 56.88%.
- GAN-Knowledge Distillation for one-stage Object Detection
    - arxiv only, 7/2019
    - the paper is not ready at all. some experiments are missing, some
      table are not fully filled.
    - the idea is interesting. It has a discriminator to predict
      whether the feature is from teacher or from student, which guides
      teh student network to mimic teacher's behavior.
- Distilling Object Detectors with Fine-grained Feature Imitation
    - CVPR 19
    - almost the same with Mimicking Very Efficient Network for
      Object Detection(cvpr17) except that, in Mimicking, it uses
      the proposal to crop the region; while 
      in this paper, it expands the region more, i.e. align the
      features within the neighbor of the target position.
      Specifically, for each ground-truth box, it first finds the
      position with highest objectness; then the objectness is
      multiplied by 0.5 as the threshold to filter all non-target region; finally the features
      within the target regions are aligned.
    - Experiment
        - it also compares the case of aligning the full feature, whose
          accuracy loses 8.9 point; while aligning the features within a
          sub region increases the accuracy by 5.2 point. The baseline
          accuracy is 62.63.
- Quantization Mimic: Towards Very Tiny CNN for Object Detection
    - ECCV 18
    - the paper is similar with mimiking features for object detection,
      where the roi feature is aligned for student's from teachers. The
      difference is that the feature is quantized before aligning, but
      there is no clue on how the network is learned, since the
      quantization operation will give 0 gradient.
- Fitnets: Hints for thin deep nets
    - ICLR 2015
    - Novelty
        - Before, the teacher-student is to transfer the knowledge
          through the soft label. The novelty here is to introduce the
          intermediate feature maps. The first step is a pre-training
          by aligning the feature map; and the second step is training
          like a usual teacher-student loss.
    - Experiment
        - on MNIST, the standard one is 1.9% (error rate); the
          knowledge distillation is 0.65%. The one with the proposed
          pre-training is 0.51%.
- Label Refinery: Improving ImageNet Classification through Label Progression
    - The teacher network passes the knowledge through soft label for
      each cropped image during the classification network training.
      The improvement here is significant, on imagenet
    - 2018 arxiv
- Mimicking Very Efficient Network for Object Detection
    - cvpr 2017
    - the knowledge is passed through the roi cropped feature for
      faster-rcnn network.
- Learning Efficient Object Detection Models with Knowledge Distillation
    - nips 2017
    - contribution:
        - add the class-aware weights on the combined classification loss. 
          The weight for all categories are the same, but different from the background
          - note, this weighting only applies to the soft loss. The
            original loss is not altered
          - compared with the non-weighted soft loss, the gain is from
            57.4 to 57.7 in VOC; from 50.8 to 51.3 in KITTI.
        - present a teacher bounded regression loss, that is, if the
          teacher's regression loss is larger, ignore the teacher's
          output and student's loss. If it is smaller, let the student
          move towards teacher's performance. This is regarded as teh
          soft regression loss, and the paper always added the normal
          regression loss.
          - compared with the non-bounded counterpart, the accuracy is
            improved from 54.6 to 55.9 in voc; from 48.5 to 50.1 in
            KITTI.
    - non-contribution, but adopt existing approaches
        - for the classification loss, it comebines the soft loss (from the
          teacher network) and the hard loss (from the ground-truth). Note,
          this is not the contribution.
        - use the feature map to align from the teacher to the student
            - one observation is that even the feature dimension is the
              same, adding a 1x1 conv layer is helpful.
                - with the adaptive layer, the accuarcy goes from 56.9
                  to 58 in voc; and 50.3 to 52.1 in KITTI.
        - with hint, the accuracy on training can be higher, which is
          kinds of counter-intuitive since it imposes more constraint
          on the features, which are not related with the ground-truth
          alignment. This might be the problem of optimization
          capability issue, where hint can help the optimization
    - experiments
        - small network is student; large network is the teacher
            - with different datasets, the accuary improvement can be
              3-4 points in terms of mAP@0.5.
            - For coco metric, the improvement is always around 1 point.
        - same network, but the network with small input image is
          student; the network with large input image is the teacher
            - the distilled student network can achieve comparable with
              the high resolution netework and is much better than the
              low resolution network.

- object detection
    - Dataset
        - Scale Match for Tiny Person Detection
            - release a dataset of TinyPerson
            - the new method is to scale extra dataset for pre-training so that
              teh size could be similar
    - Teacher student
        - Mimicking Very Efficient Network for Object Detection
            - student network extracts the region proposal, which is used to
              extract the features from the student network and the teacher
              netowork. The idea is to align the extracted features.
                - The student network can also receive half-sized image as input. The paper discussed this optition, but there is no experiment about this, which is strange.
            - cvpr 17
    - Domain adaptation
        - Few-shot Adaptive Faster R-CNN
            - CVPR 19
            - github was set by the author, but no code is shared (1/30/2020)
        - SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses
            - on arxiv 11/29/2019
            - cmu
            - setting
                - the source domain has full annotations (label and bounding
                  boxes)
                - the target domain has no labels at all. No image-level labels
            - novelty
                - on the backbone, 3 branches are inserted with the domain
                  classifier, so that the features could be in the same domain
                  with reverse gradient policy
                    - the loss can be cross entropy, weighted loss, focal loss.
                      The author studied the accuracy with different losses
                - context feature is extracted from the branches to combine
                  with the region-level features. Thus, each region-level
                  feature contains an image-level feature, which are finally
                  used for classification and regression
    - application
        - Model Adaption Object Detection System for Robot
        - EdgeNet: Balancing Accuracy and Performance for Edge-based Convolutional Neural Network Object Detectors
            - on arxiv 11/4/2019
            - looks like a hardware related paper. Not interested
        - RoIMix: Proposal-Fusion among Multiple Images for Underwater Object Detection
            - arxiv 11/8/2019
            - Peking University
            - Contribution
                - The region proposal is mixed up by another region proposal
                    - the mix-up is performed on the features, not on coordinates.
                    - the mix-up is performed by a linear combination
                    - the label is not changed by the second region proposal. Thus
                      the weight for the current poposal is always larger than that
                      for the second proposal by performing max operation
                      (max(lambda, 1 - lambda))
                    - the benefit is to mimic overlapping for underwater dataset
                      because it could be transparent in this case
    - Small network
        - Localization-aware Channel Pruning for Object Detection
            - arxiv 11/21/2019
            - Huazhong Univerisity
            - reduce network parameters.
    - Network architecture
        - Rethinking Classification and Localization for Object Detection
            - arxiv 12/2019
            - key idea
                - use two networks in box head to process the roi feature. One
                  is based on fully connected layers; the other is based on
                  conv layers. One approach is to add classification and
                  localization loss on each head; the other is to add
                  classification loss on the fully-connected head and the other
                  is to add localization loss on the conv layer
            - experiment
                - one loss on each head
                    - 37.3 -> 38.8
        - IoU-aware Single-stage Object Detector for Accurate Localization
            - arxiv 2019/12
            - claimed novelty
                - predict the IoU of the predicted box and the ground truth.
                - the final score is the IoU multiplied by the classification
                  prediciton
                - this is the same as what Yolo V2 does.
            - best acc is 40.6 on coco -> X101-FPN-RetinaNet
        - Learning Rich Features at High-Speed for Single-Shot Object Detection
            - iccv/2019
            - novelty
                - downsample the raw image to multiple sub scales and fuse it
                  with the feature maps from different levels
            - highest acc is 37.3 on coco with 32 ms for each image on Titan X
        - Guided Attention Network for Object Detection and Counting on Drones
            - 2019/9
            - novelty
                - 4 feature maps. 1/2, 1/4, 1/8, 1/16 for small scale
                - the loss is applied on the feature map of 1/2, not on all
                  these 4 maps
                - a component of background attention module, which is used to
                  fuse the feature map from higher level (smaller size) with
                  the current feature map.
            - the best approach on CARPK is 90.2.
        - Attentional Network for Visual Object Detection
            - not that interesting.
            - 2017
            - no experiment on coco
            - Some recurrent network with reinforcement learning
        - Objects as Points
            - 2019/4 in arxiv.
            - Novelty
                - In Yolov2, the single feature map is with stride of 32. In
                  this paper, it is 4.
                - The objectness is learned with focal loss. The objectness is
                  called centerness here. 
                - the objectness is class-specific. That is, if we have 80
                  classes, we have 80 feature maps. In YoloV2, it is
                  class-agnositic
                - cneter offset is predicted as a class-agnositic way. In
                  YoloV2, it is class specific
                - box size is predicted as class-agnositc way, which is the
                  same as YoloV2.
            - experiment
                - the highest accuracy it can achieve is 42.1 with hourglass-104 as the backbone

        - Improving Object Detection with Inverted Attention
            - on arxiv 3/28/2019 
            - contribution
                - the feature map is re-weighted by its inversed gradient
                    - the intuition is that the gradient is high on the most
                      discriminative regions. If we reverse it, we can focus on
                      less discriminative regions. Thus, the idea is try to
                      focus on the whole region part rather than the most
                      discriminative regions. However, it is not straigtforward
                      to conclude the accuracy would be better. Meanwhile, in
                      the experiment, only 20% features are re-weighted.
        - Enriched Feature Guided Refinement Network for Object Detection
            - ICCV 19
            - https://github.com/Ranchentx/EFGRNet
            - Tianjin University
            - Contribution
                - a framework to enchance the features used for prediction at each
                  prediction layer
                  - a feature enchancemenet module
                    - the input image is first downsampled to 1/8. Then, it is fed
                      to a convolutional network as the enchanced feature.
                      - the netework here contains dilated=1, 2, 4, conv layers to
                        contain more contextual features.
                    - acc is improved from 77.2 to 79.4 on voc
                    - if dilation is 1, the acc is 78.7. Change one as dilation =
                      2, the acc is 79.0, with another as dilation=4, the acc is
                      79.4.
                  - Feature guided refinement module
                      - from the enchanced feature, it predicts an objectness for
                        each anchor. Then, sum up all the objectness for all
                        different anchor shapes at the same spatial location, as
                        the attention. The final feature is the original enchanced
                        feature with original feature multiplied by the attention.
                      - Before doing the final prediction, it uses a deconv layer
                        to filter the features. The deconv offset comes from the
                        offset prediction (bounding box regression).
                      - acc is improved from 77.2 to 81.0
                   - with the two modules, the acc is improved from 77.2 to 81.4
            - Experiment
                - on voc, the baseline is 77.2, and the approach improves by 4.1
                  point.
                  - But the improved solution has lots of more parameters and
                    computations. The comparision might come from more parameters
                    and computations. It is unclear if the accuracy is higher with
                    similar computations
                - on coco, the baseline is 20ms with acc = 25.1. The improved one
                  is 21ms with accc = 33.2, which looks pretty promising.
                  - However, it is not clear if the training logic is the same,
                    e.g. the number of iterations.
        - EfficientDet: Scalable and Efficient Object Detection
        - Learning Spatial Fusion for Single-Shot Object Detection
        - Beihang University
        - Strong baseline
            - improve yolov3 with existing approaches
                - bag of tricks (33.0 to 37.2)
                    - mixup algorithm
                    - cosine learning rate scheduling
                    - sync bn
                - add one anchor-free branch together with anchor-based
                  branched
                - add anchor guiding mechanism (37.2 to 38.2)
                    - Region proposal by guided anchoring
                        - CVPR19
                - add IoU loss (37.2 to 37.6)
                - final 37.2 -> 38.8
        - contribution
            - adaptive fusion, i.e. fuse the three feature maps with
              adaptive weights. Each spatial position has a different
              weight. The weights on the same location but from different
              feature maps are summed as 1.
        - other details
            - data augmentation: 320 to 608
            - NMS: 0.6
            - 300 epochs
            - cosine learning rate from 0.001 to 1e-5
            - weight decay 5e-4
            - turn off mixup augmentation for the last 30 epochs
            - ignore the adjacent negative samples at the same location
              with the positives. epsilon: ignore region ratio
                - epsilon = 0.2 -> 38.8 -> 39.1
                - epsilon = 0.5 -> 38.8 -> 37.5
            - 38.8 -> 40.6 by the proposed adaptive fusion method
            - other fusion method
                - if we use sum as fusion method: 38.8 -> 39.3
                - if we use concat as fusion method: 38.8 -> 39.5. The number
                  of parameters are not disclosed
            - improve retinaNet with the fusion method from 35.9 to 37.4
              with R50 and from 39.1 to 40.1 with R101. Note, the
              comparision with sum and concatenation is not disclosed
        - official code release: https://github.com/ruinmessi/ASFF

