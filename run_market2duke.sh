## step 1 Source-domain pre-training
# Duke to Market
# python source_pretrain.py -ds market -dt duke --data-dir /hdd4/data/reid_training/ --logs-dir logs/market2duke/source_pretraining -b 64

## step 2 Target-domain fine-tuning
# Duke to Market
# python target_train.py -dt duke --data-dir /hdd4/data/reid_training/ --logs-dir logs/market2duke/target_fine_tuning --initial-weights logs/market2duke/source_pretraining -b 128

## step 3 Evaluate in the target domain
# Duke to Market
# python model_test.py -dt duke --data-dir /hdd4/data/reid_training/ --resume logs/market2duke/target_fine_tuning/model_best.pth.tar 