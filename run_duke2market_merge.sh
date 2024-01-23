## step 1 Source-domain pre-training
# duke-to-market merge
<<<<<<< HEAD
python source_pretrain.py -ds duke -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --logs-dir logs/duke2market_merge/source_pretraining -b 64

## step 2 Target-domain fine-tuning
# duke-to-market merge
python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --logs-dir logs/duke2market_merge/target_fine_tuning --initial-weights logs/duke2market_merge/source_pretraining -b 64
=======
CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds duke -dt market --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_merge/source_pretraining -b 128

## step 2 Target-domain fine-tuning
# duke-to-market merge
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt market --data-dir /home/ccvn/Workspace/trinh/data/reid --logs-dir logs/duke2market_merge/target_fine_tuning --initial-weights logs/duke2market_merge/source_pretraining -b 128
>>>>>>> 91bd37badabcfa7fa36c54f155222beb91f8f7f7

## step 3 Evaluate in the target domain
# duke-to-market merge
# CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt market --data-dir /home/ccvn/Workspace/trinh/data/reid --resume logs/duke2market_merge/target_fine_tuning/model_best.pth.tar