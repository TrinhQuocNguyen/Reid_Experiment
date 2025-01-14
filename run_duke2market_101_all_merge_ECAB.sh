## step 1 Source-domain pre-training
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds duke -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_101_all_merge_ECAB_Z/source_pretraining -b 128 --arch resnet101_source

## step 2 Target-domain fine-tuning
# for example, duke-to-market
CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_101_all_merge_ECAB_Z/target_fine_tuning_ECAB_BMFN_Local_Loss --initial-weights logs/duke2market_101_all_merge_ECAB_Z/source_pretraining -b 128 --num-clusters 700 --arch resnet101

## step 3 Evaluate in the target domain
# for example, duke-to-market
CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_101_all_merge_ECAB_Z/target_fine_tuning_ECAB_BMFN/model_best.pth.tar --num-classes 700 --arch resnet101