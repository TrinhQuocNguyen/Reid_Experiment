# CORE-ReID
CORE-ReID V2 : Advancing the Domain Adaptation for Object Re-Identification with Optimized Training and Ensemble Fusion

Our project page: https://trinhquocnguyen.github.io/core-reid-v2-homepage/

**[2024/12/01: Good News!]** ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)    

* We have developed the second version of CORE-ReID, the performance is much better, please stay in tune.

## TODO
- [ ] Explain step by step how to run the source code with tutorial videos
- [X] Update Readme.md file
- [X] Fix the absolute path
- [X] Initialize the source code

## Updates
- **[2024/12/01: Good News!]** ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)
    * We have developed the second version of CORE-ReID, the performance is much better, please stay in tune.
- **[2024/07/14: Source code is released!]** ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)
    * The first version of source code has been initialized.
- **[2024/06/03: CORE-ReID  is published by MDPI!]**
    * The paper of CORE-ReID is publised by MDPI: https://www.mdpi.com/2674-113X/3/2/12 
- **[2024/05/27: CORE-ReID is accepted by MDPI!]**
    * The paper of CORE-ReID is accepted by MDPI.

## (I) Installation
This codebase has been developed with python version 3.8, PyTorch version 1.8.1, CUDA 12.1 and torchvision 0.9.1
- Installation
    - Run: ```conda env create -f core_reid_v2.yml```
- We used serveral servers to train the models:
    - NVIDIA Quadro RTX 8000 x 2


## (II) Training 
### 1. Train CycleGAN models
Comming...
### 2. Generate the training datataset
Comming...
### 3. Train the ReID model - Step 1: Pretraining on Source Domain
For example, CUKH03 => MARKET1501: 
```sh scripts/run_cuhk03np2market.sh```
### 4. Train the ReID model - Step 2: Fine-Tuning on Target Domain
Comming...
### 5. Train the ReID model - Step 3: Evaluation on Target Domain
Comming...

## Acknowledgement
Thank you for great works below:
- [open-reid](https://github.com/Cysu/open-reid)
- [MEB-Net](https://github.com/YunpengZhai/MEB-Net)
- [LF2](https://github.com/DJEddyking/LF2)
- [CBAM](https://github.com/luuuyi/CBAM.PyTorch)
