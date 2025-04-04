################################## //RESNET 101// ################################## 
## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet101/700_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
                       --num-clusters 700 --arch resnet101

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet101/500_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
                       --num-clusters 500 --arch resnet101