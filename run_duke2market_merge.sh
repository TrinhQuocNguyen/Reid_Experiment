## step 1 Source-domain pre-training
# duke-to-market merge
python source_pretrain.py -ds duke -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --logs-dir logs/duke2market_merge/source_pretraining -b 128

## step 2 Target-domain fine-tuning
# duke-to-market merge
python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --logs-dir logs/duke2market_merge/target_fine_tuning --initial-weights logs/duke2market_merge/source_pretraining -b 128

## step 3 Evaluate in the target domain
# duke-to-market merge
python model_test.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/duke2market_merge/target_fine_tuning/model_best.pth.tar