# Yolov5 + Deep Sort with PyTorch





<div align="center">
<p>
<img src="MOT16_eval/track_pedestrians.gif" width="400"/> <img src="MOT16_eval/track_all.gif" width="400"/> 
</p>
<br>
<div>
<a href="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/actions"><img src="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<br>  
<a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 
</div>

</div>


## Introduction

This repository contains a two-stage-tracker. The detections generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to a [Deep Sort algorithm](https://github.com/ZQPei/deep_sort_pytorch) which tracks the objects. It can track any object that your Yolov5 model was trained to detect.


## Tutorials

* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;
* [Yolov5 deep_sort pytorch evaluation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)&nbsp;



## Before you run the tracker

1. Clone the repository recursively:

`git clone https://github.com/2021-SKKU-AUTOSEMANTICS-SANHAK/Yolov5_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt] dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`

3. Download re-identification weights 

`https://drive.google.com/drive/folders/1o5cphaQHpOlauHoPnbwdkmsiYSfZAvR0?usp=sharing`

model_data/를 Yolov5_DeepSort_Pytorch/의 바로 아래에 위치시킨다

4. Download yolov5 weights

`https://github.com/ultralytics/yolov5/releases`

위 링크에서 yolov5 weight를 다운로드받고, yolov5/weights/ 에 위치시킨다. ex) yolov5/weights/yolov5x.pt

5. google cloud platform key를 추가한다(local input video 사용 시 필요 없음, realtime==1일때만 사용됨)

Documentation의 setting 부분에서 key가 위치한 주소를 찾을 수 있습니다.. key는 3번과 마찬가지로 최상단 폴더에 위치시켜주세요



## Re-ID Models

| Model Name | Pretrained Model Name | Train Dataset | Test Dataset | Loss | Learning Rate | Epoch | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - | OSNet x1.0 | market1501 | market1501 | - | - | - | 84.90% |
| - | OSNet x1.0 | SKKU | SKKU | - | - | - | 88.90% |
| plr_osnet.pth (market1501) | PLR-OSNet | market1501 | market1501 | - | - | - | 88.90% |
| plr_osnet.pth (SKKU) | PLR-OSNet | SKKU | SKKU | - | - | - | 85.90% |
| resnet50.pth.tar-90 | Resnet-50 | CUHK03 | CUHK03 | softmax | 0.0003 | 90 | 44.30% |
| market1501.pth.tar-120 | PLR-OSNet | market1501 | market1501 | softmax | 0.000035 | 120 | 89.20% |
| market1501+cuhksysu.pth.tar-140 | OSNet x1.0 | market1501 & CUHKSYSU | market1501 | softmax | 0.0015 | 140 | 83.20% |
| osnet.pth.tar-80 | OSNet x1.0 | SKKU | SKKU | softmax | 0.0015 | 80 | 88.90% |
| plr_osnet.pth.tar-110 | PLR-OSNet | SKKU | SKKU | triplet | 0.000045 | 110 | 87.10% |
| lup_moco_r50.pth | MOCO v2 | market1501 | market1501 | - | - | - | 91.12% |




