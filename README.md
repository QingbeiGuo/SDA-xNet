# SDA-xNet: Selective Depth Attention Networks for Adaptive Multi-scale Feature Representation

## Introduction

Existing multi-scale solutions lead to a risk of just increasing the receptive field sizes while neglecting small receptive fields.
Thus, it is a challenging problem to effectively construct adaptive neural networks for recognizing various spatial-scale objects.
To tackle this issue, we first introduce a new attention dimension, i.e., depth, in addition to existing attention dimensions such as channel, spatial, and branch, and present a novel selective depth attention network to symmetrically handle multi-scale objects in various vision tasks.
Specifically, the blocks within each stage of a given neural network, i.e., ResNet, output hierarchical feature maps sharing the same resolution but with different receptive field sizes.
Based on this structural property, we design a stage-wise building module, namely SDA, which includes a trunk branch and a SE-like attention branch.
The block outputs of the trunk branch are fused to globally guide their depth attention allocation through the attention branch.
According to the proposed attention mechanism, we can dynamically select different depth features, which contributes to adaptively adjusting the receptive field sizes for the variable-sized input objects.
In this way, the cross-block information interaction leads to a long-range dependency along the depth direction.
Compared with other multi-scale approaches, our SDA method combines multiple receptive fields from previous blocks into the stage output, thus offering a wider and richer range of effective receptive fields.
Moreover, our method can be served as a pluggable module to other multi-scale networks as well as attention networks, coined as SDA-$x$Net.
Their combination further extends the range of the effective receptive fields towards small receptive fields, enabling interpretable neural networks.
Extensive experiments demonstrate that the proposed SDA method achieves state-of-the-art (SOTA) performance, outperforming other multi-scale and attention counterparts on numerous computer vision tasks, e.g., image classification, object detection, and instance segmentation.
Our source code is available at https://github.com/QingbeiGuo/SDA-xNet.git.

This project is a pytorch implementation, aiming to compressing and accelerating deep convolutional neural networks. 

### What we are doing and going to do

- [x] Support pytorch-1.12.

## Classification

We benchmark our code thoroughly on imagenet-1K for classification, using different network architectures. Below are the results:

1) Classification Comparison among several SOTA multi-scale methods on ImageNet-1K.

Method   | Backbone Model | Multi-scale | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|---------|---------|--------|-------|-----------|-----------
ResNet-50                |  \multirow{7}{*}{ResNet}        | multi-scale |   25.56  |  4.12  |  75.20  |  92.52
bL-ResNet                |                                 | multi-scale |   26.69  |  2.85  |  77.31  |  -
ScaleNet                 |                                 | multi-scale |   31.48  |  3.82  |  77.98  |  93.95
EPSANet                  |                                 | multi-scale |   22.59  |  3.60  |  77.49  |  93.54
PyConv                   |                                 | multi-scale |   24.85  |  3.88  |  77.88  |  93.80
Res2Net                  |                                 | multi-scale |   25.70  |  4.26  |  77.99  |  93.85
SDA-ResNet-86 (ours)     |                                 | multi-scale |   27.22  |  3.88  |  \textbf{78.76}  |  \textbf{94.37}
---------|---------|---------|--------|-------|-----------|-----------
EPSANet-101              |  \multirow{3}{*}{EPSANet}       | multi-scale |   38.90  |  6.82  |  78.43  |  94.11
EPSANet-86 (our impl.)   |                                 | multi-scale |   36.67  |  5.84  |  77.71  |  93.83
SDA-EPSANet-86 (ours)    |                                 | multi-scale |   39.45  |  5.85  |  \textbf{78.80}  |  \textbf{94.34}
---------|---------|---------|--------|-------|-----------|-----------
PyConv-101               |  \multirow{3}{*}{PyConv}        | multi-scale |   44.63  |  8.42  |  79.22  |  94.43
PyConv-86 (our impl.)    |                                 | multi-scale |   40.55  |  6.22  |  78.63  |  94.26
SDA-PyConv-86 (ours)     |                                 | multi-scale |   43.33  |  6.22  |  \textbf{79.27}  |  \textbf{94.65}
---------|---------|---------|--------|-------|-----------|-----------
Res2Net-101              |  \multirow{3}{*}{Res2Net}       | multi-scale |   45.21  |  8.10  |  79.19  |  94.43
Res2Net-86 (our impl.)   |                                 | multi-scale |   42.47  |  6.94  |  78.63  |  94.21
SDA-Res2Net-86 (ours)    |                                 | multi-scale |   45.26  |  6.95  |  \textbf{79.36}  |  \textbf{94.70}

2) Classification Comparison among several SOTA attention methods on ImageNet-1K.

Method   | Backbone Model | Attention Dimension | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|---------|---------|--------|-------|-----------|-----------
ResNet-50                |  \multirow{13}{*}{ResNet}      | -                     |   25.56  |  4.12  |  75.20  |  92.52
SENet                    |                                | channel               |   28.09  |  4.09  |  76.71  |  93.38
ECANet                   |                                | channel               |   25.56  |  4.09  |  77.48  |  93.68
GSoP                     |                                | channel               |   28.05  |  6.18  |  77.68  |  93.98
FcaNet                   |                                | channel               |   28.07  |  4.09  |  78.52  |  94.14
A$^2$Net                 |                                | spatial               |   33.00  |  6.50  |  77.00  |  93.50
AANet                    |                                | spatial               |   25.80  |  4.15  |  77.70  |  93.80
GCNet                    |                                | spatial               |   28.11  |  4.13  |  77.70  |  93.66
BAM                      |                                | channel+spatial       |   25.92  |  4.17  |  75.98  |  92.82
CBAM                     |                                | channel+spatial       |   30.62  |  4.10  |  77.34  |  93.69
SANet                    |                                | channel+spatial       |   25.56  |  4.09  |  77.72  |  93.80
EPSANet                  |                                | branch                |   22.59  |  3.60  |  77.49  |  93.54
SDA-ResNet-86 (ours)     |                                | depth                 |   27.22  |  3.88  |  \textbf{78.76}  |  \textbf{94.37}
---------|---------|---------|--------|-------|-----------|-----------
SENet-101                |  \multirow{3}{*}{SENet}        | channel               |   49.33  |  7.81  |  77.62  |  93.93
SENet-86 (our impl.)     |                                | channel               |   46.35  |  6.71  |  78.29  |  93.92
SDA-SENet-86 (ours)      |                                | depth+channel         |   49.13  |  6.72  |  \textbf{78.95}  |  \textbf{94.50}
---------|---------|---------|--------|-------|-----------|-----------
CBAM-101                 |  \multirow{3}{*}{CBAM}         | channel+spatial       |   54.04  |  7.81  |  78.49  |  94.31
CBAM-86 (our impl.)      |                                | channel+spatial       |   50.75  |  6.71  |  78.36  |  94.09
SDA-CBAM-86 (ours)       |                                | depth+channel+spatial |   53.53  |  6.72  |  \textbf{79.05}  |  \textbf{94.44}
---------|---------|---------|--------|-------|-----------|-----------
EPSANet-101              |  \multirow{3}{*}{EPSANet}      | branch                |   38.90  |  6.82  |  78.43  |  94.11
EPSANet-86 (our impl.)   |                                | branch                |   36.67  |  5.84  |  77.71  |  93.83
SDA-EPSANet-86 (ours)    |                                | depth+branch          |   39.45  |  5.85  |  \textbf{78.80}  |  \textbf{94.34}


## Object Detection

Object detection results of different methods on COCO val 2017.

Detector    |Model    | Params | FLOPs | AP (%) | AP_50 (%) | AP_75 (%) | AP_S (%) | AP_M (%) | AP_L (%)     
---------|---------|--------|-------|-----------|-----------|-----------|-----------|-----------|-----------
\multirow{11}{*}{Faster-RCNN} |ResNet-50                    |  41.53  |  207.07  | 36.4 | 58.2 | 39.2 | 21.8 | 40.0 | 46.2
                              |EPSANet                      |  38.56  |  197.07  | 39.2 | 60.3 | 42.3 | 22.8 | 42.4 | 51.1
                              |SENet                        |  44.02  |  207.18  | 37.7 | 60.1 | 40.9 | 22.9 | 41.9 | 48.2
                              |ECANet                       |  41.53  |  207.18  | 38.0 | 60.6 | 40.9 | 23.4 | 42.1 | 48.0
                              |FcaNet                       |  44.02  |  215.63  | 39.0 | \textbf{61.1} | 42.3 | \textbf{23.7} | 42.8 | 49.6
                              |Res2Net                      |  41.67  |  215.52  | 33.7 | 53.6 | -    | 14.0 | 38.3 | 51.1
                              |SDA-ResNet-86 (ours)         |  43.11  |  202.93  | \textbf{39.8} | 60.7 | \textbf{43.2} | 22.9 | \textbf{43.9} | \textbf{51.4}
                              \cline{2-10}%\specialrule{0.08em}{0.5pt}{0.5pt}
                              |SENet-101                    |  65.24  |  295.58  | 39.6 | 62.0 | 43.1 | 23.7 | 44.0 | 51.4
                              |SDA-SENet-86 (ours)                        |  64.81  |  260.93  | \textbf{41.0} | \textbf{62.1} | \textbf{44.5} | \textbf{23.9} | \textbf{44.9} | \textbf{53.4}
                              \cline{2-10}%\specialrule{0.08em}{0.5pt}{0.5pt}
                              |Res2Net-101                  |  61.18  |  293.68  | -    | -    | -    | -    | -    | -
                              |SDA-Res2Net-86 (ours)        |  60.98  |  265.70  | 42.0 | 62.7 | 45.5 | 24.1 | 45.8 | 55.0
---------|---------|--------|-------|-----------|-----------|-----------|-----------|-----------|-----------
\multirow{11}{*}{Mask-RCNN}   |ResNet-50                    |  44.17  |  261.81  | 37.2 | 58.9 | 40.3 | 22.2 | 40.7 | 48.0
                              |EPSANet                      |  41.20  |  248.53  | 40.0 | 60.9 | 43.3 | 22.3 | 43.2 | 52.8
                              |SENet                        |  46.66  |  261.93  | 38.7 | 60.9 | 42.1 | 23.4 | 42.7 | 50.0
                              |ECANet                       |  44.17  |  261.93  | 39.0 | 61.3 | 42.1 | \textbf{24.2} | 42.8 | 49.9
                              |FcaNet                       |  46.66  |  261.93  | 40.3 | \textbf{62.0} | 44.1 | 25.2 | 43.9 | 52.0
                              |Res2Net                      |  44.31  |  268.59  | 39.6 | 60.9 | 43.1 | 22.0 | 42.3 | 52.8
                              |SDA-ResNet-86 (ours)         |  45.75  |  256.01  | \textbf{40.6} | 61.1 | \textbf{44.4} | 23.4 | \textbf{44.3} | \textbf{53.8}
                              \cline{2-10}%\specialrule{0.08em}{0.5pt}{0.5pt}
                              |SENet-101                    |  67.88  |  348.65  | 40.7 | 62.5 | 44.3 | 23.9 | 45.2 | 52.8
                              |SDA-SENet-86 (ours)          |  67.46  |  314.00  | \textbf{41.5} | \textbf{62.8} | \textbf{45.2} | \textbf{24.8} | \textbf{45.6} | \textbf{53.7}
                              \cline{2-10}%\specialrule{0.08em}{0.5pt}{0.5pt}
                              |Res2Net-101                  |  63.82  |  346.74  | 41.8 | 62.6 | 45.6 | 23.4 | 45.5 | 55.6
                              |SDA-Res2Net-86 (ours)        |  63.62  |  318.78  | \textbf{42.3} | \textbf{62.9} | \textbf{46.4} | \textbf{24.5} | \textbf{46.2} | \textbf{55.9}


## Instance Segmentation

Instance segmentation results of different methods on COCO val 2017.

Detector    |Model    | Params | FLOPs | AP (%) | AP_50 (%) | AP_75 (%) | AP_S (%) | AP_M (%) | AP_L (%)     
---------|---------|--------|-------|-----------|-----------|-----------|-----------|-----------|-----------
Mask-RCNN   |ResNet-50                   |  44.17  |  261.81  | 34.1 | 55.5 | 36.2 | 16.1 | 36.7 | 50.0
Mask-RCNN   |EPSANet                     |  41.20  |  248.53  | 35.9 | 57.7 | 38.1 | \textbf{18.5} | 38.8 | 49.2
Mask-RCNN   |SENet                       |  46.66  |  261.93  | 35.4 | 57.4 | 37.8 | 17.1 | 38.6 | 51.8
Mask-RCNN   |ECANet                      |  44.17  |  261.93  | 35.6 | 58.1 | 37.7 | 17.6 | 39.0 | 51.8
Mask-RCNN   |FcaNet                      |  46.66  |  261.93  | 36.2 | \textbf{58.6} | 38.1 | -    | -    | -
Mask-RCNN   |Res2Net                     |  44.31  |  268.59  | 35.6 | 57.6 | 37.6 | 15.7 | 37.9 | \textbf{53.7}
Mask-RCNN   |SDA-ResNet-86 (ours)        |  45.75  |  256.01  | \textbf{36.5} | 58.2 | \textbf{39.0} | 16.9 | \textbf{39.5} | 53.5                  
---------|---------|--------|-------|-----------|-----------|-----------|-----------|-----------|-----------
Mask-RCNN   |SENet-101                   |  67.88  |  348.65  | 36.8 | 59.3 | 39.2 | 17.2 | 40.3 | \textbf{53.6}
Mask-RCNN   |SDA-SENet-86 (ours)         |  67.46  |  314.00  | \textbf{37.0} | \textbf{59.6} | \textbf{39.3} | \textbf{18.2} | \textbf{40.6} | 53.4
---------|---------|--------|-------|-----------|-----------|-----------|-----------|-----------|-----------
Mask-RCNN   |Res2Net-101                 |  63.82  |  346.74  | 37.1 | 59.4 | 39.4 | 16.6 | 40.0 | \textbf{55.6}\\
Mask-RCNN   |SDA-Res2Net-86 (ours)       |  63.62  |  318.78  | \textbf{37.9} | \textbf{59.8} | \textbf{40.7} | \textbf{18.3} | \textbf{41.0} | 55.2


## Training for Classification

python train.py 

## Training for Object Detection

CUDA_VISILE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_1x_coco_sdanet.py 4 --work-dir work_dirs/faster_rcnn_r50_1x_coco_sdanet
CUDA_VISILE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_1x_coco_sdanet.py 4 --work-dir work_dirs/mask_rcnn_r50_1x_coco_sdanet

## Training for Instance Segmentation

CUDA_VISILE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_1x_coco_sdanet.py 4 --work-dir work_dirs/mask_rcnn_r50_1x_coco_sdanet

## Authorship

This project is contributed by [Qingbei Guo](https://github.com/QingbeiGuo).

## Citation
