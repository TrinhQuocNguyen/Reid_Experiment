
## Step 1 Source-domain pre-training
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                           --logs-dir logs/cuhk03np2market/resnet34/source_pretraining -b 128 --arch resnet34_source \
#                           --epochs 2 --iters 2 --eval-step 1
# ## Step 2 Target-domain fine-tuning                        
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                        --logs-dir logs/cuhk03np2market/resnet34/target_fine_tuning_900_global_FUSE \
#                        --initial-weights logs/cuhk03np2market/resnet34/source_pretraining -b 128 \
#                        --num-clusters 900 --arch resnet34 \
#                        --epochs 2 --iters 2 --eval-step 1


################################## //RESNET 18// ################################## 
## Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/cuhk03np2market/resnet18/source_pretraining -b 128 --arch resnet18_source

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet18/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet18/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet18

################################## //RESNET 34// ################################## 
## Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/cuhk03np2market/resnet34/source_pretraining -b 128 --arch resnet34_source

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet34/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet34/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet34

################################## //RESNET 50// ################################## 
## Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/cuhk03np2market/resnet50/source_pretraining -b 128 --arch resnet50_source

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet50/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet50/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet50

################################## //RESNET 101// ################################## 
## Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/cuhk03np2market/resnet101/source_pretraining -b 128 --arch resnet101_source

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet101/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet101

################################## //RESNET 150// ################################## 
## Step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                        --logs-dir logs/cuhk03np2market/resnet150/source_pretraining -b 128 --arch resnet150_source

## Step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                       --logs-dir logs/cuhk03np2market/resnet150/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet150/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet150