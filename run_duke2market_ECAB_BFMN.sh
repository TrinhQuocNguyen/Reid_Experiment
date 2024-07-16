## step 1 Source-domain pre-training
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds duke -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid  --logs-dir logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --arch resnet101_source

## step 2 Target-domain fine-tuning
# for example, duke-to-market
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_ECAB_BFMN/target_fine_tuning_700_global_ECAB --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --num-clusters 700 --arch resnet101


## step 3 Evaluate in the target domain
# for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_700/model_best.pth.tar --num-classes 700 --arch resnet101 


# ## step 2 Target-domain fine-tuning
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_ECAB_BFMN/target_fine_tuning_500 --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --num-clusters 500 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_500/model_best.pth.tar --num-classes 500 --arch resnet101 


# ## step 2 Target-domain fine-tuning
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_ECAB_BFMN/target_fine_tuning_900 --initial-weights logs/duke2market_ECAB_BFMN/source_pretraining -b 128 --num-clusters 900 --arch resnet101

# ## step 3 Evaluate in the target domain
# # for example, duke-to-market
# CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_ECAB_BFMN/target_fine_tuning_900/model_best.pth.tar --num-classes 900 --arch resnet101 
