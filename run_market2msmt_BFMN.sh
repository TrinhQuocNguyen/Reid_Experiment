
# ## step 2 Target-domain fine-tuning
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2msmt_BFMN/target_fine_tuning_2500 --initial-weights logs/market2msmt_ECAB_BFMN/source_pretraining -b 128 --num-clusters 2500 --arch resnet101

# ## step 3 Evaluate in the target domain
CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/market2msmt_BFMN/target_fine_tuning_2500/model_best.pth.tar --num-classes 2500 --arch resnet101 

