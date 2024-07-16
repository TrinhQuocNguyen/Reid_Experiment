# ## step 2 Target-domain fine-tuning
# # Market to Duke
python target_train.py -dt msmt17 --data-dir /mnt/AIProjects/trinh/DATA/reid \
                    --logs-dir logs/market2msmt_ECAB_BFMN/target_fine_tuning_Global_ECAB_BFMN \
                    --initial-weights logs/market2msmt_ECAB_BFMN/source_pretraining -b 128 --num-clusters 2500 --arch resnet101

## step 3 Evaluate in the target domain
# Market to Duke
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt msmt17 --data-dir /home/ccvn/Workspace/trinh/data/reid \
#                     --resume logs/market2msmt_101_all_merge_ECAB/target_fine_tuning/model_best.pth.tar \
#                     --num-classes 3000 --arch resnet101