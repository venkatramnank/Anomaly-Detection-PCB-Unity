# PaDiM-Anomaly-Detection-PCB-Nokia
This is an implementation of the paper [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).   

This code is heavily borrowed from the unofficial implementation by [xiahaifeng1995](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) and the efficient net version implementation by [yougjae](https://github.com/youngjae-avikus/PaDiM-EfficientNet/tree/master)

## Requirements
* python >= 3.7
* pytorch >= 1.5
* tqdm
* sklearn
* matplotlib
* [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## Datasets
For initial testing , the  MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/) is used. 

Custom PCB data created using Unity.

## Running 

In order to train the Efficient net model
```python
 python efficient_main.py -d [dataset location] -s [results storage directory] --training 
 ```

## Reference
[1] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. https://arxiv.org/pdf/2011.08785

[2] https://github.com/byungjae89/SPADE-pytorch

[3] https://github.com/byungjae89/MahalanobisAD-pytorch
