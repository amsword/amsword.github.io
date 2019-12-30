---
layout: post
comments: true
title: Paper Reading
---

- Guided Attention Network for Object Detection and Counting on Drones
- Attentional Network for Visual Object Detection
- Improving Object Detection with Inverted Attention
- SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses
- image classification
    - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
        - ICML 2019
        - contribution
            - scale the network width (channels), depth, and input image size at the same time
        - others
            - apply an existing work (same author) to search a network with fewer target flops and 
              use flops as the cost measurement rather than hardware cost
                - MnasNet: Platform-aware neural architecture search for mobile
- network architecture search
    - MnasNet: Platform-aware neural architecture search for mobile
- object detection
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
    - Localization-aware Channel Pruning for Object Detection
        - arxiv 11/21/2019
        - Huazhong Univerisity
        - reduce network parameters.
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

