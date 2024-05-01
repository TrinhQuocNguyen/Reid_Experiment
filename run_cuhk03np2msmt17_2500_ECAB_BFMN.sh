## step 1 Source-domain pre-training
CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds cuhk03np -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/cuhk03np2msmt_2500_ECAB_BFMN/source_pretraining -b 128 --arch resnet101_source



## step 2 Target-domain fine-tuning
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning --initial-weights logs/cuhk03np2msmt_2500_ECAB_BFMN/source_pretraining -b 128 --num-clusters 2500 --arch resnet101

## step 3 Evaluate in the target domain
CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning/model_best.pth.tar --num-classes 2500 --arch resnet101




# ## step 2 Target-domain fine-tuning
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning_3000 --initial-weights logs/cuhk03np2msmt_2500_ECAB_BFMN/source_pretraining -b 128 --num-clusters 3000 --arch resnet101

# ## step 3 Evaluate in the target domain
# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning_3000/model_best.pth.tar --num-classes 3000 --arch resnet101

# ## step 2 Target-domain fine-tuning
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning_3500 --initial-weights logs/cuhk03np2msmt_2500_ECAB_BFMN/source_pretraining -b 128 --num-clusters 3500 --arch resnet101

# ## step 3 Evaluate in the target domain
# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/cuhk03np2msmt_2500_ECAB_BFMN/target_fine_tuning_3500/model_best.pth.tar --num-classes 3500 --arch resnet101

