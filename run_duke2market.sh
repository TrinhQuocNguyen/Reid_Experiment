## step 1 Source-domain pre-training
# for example, duke-to-market
# export CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=1 python source_pretrain.py -ds duke -dt market --data-dir /data/reid/ --logs-dir logs/duke2market/source_pretraining -b 64

## step 2 Target-domain fine-tuning
# for example, duke-to-market
# python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --logs-dir logs/duke2market/target_fine_tuning --initial-weights logs/duke2market/source_pretraining -b 128

## step 3 Evaluate in the target domain
# for example, duke-to-market
# python model_test.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid --resume logs/duke2market/target_fine_tuning/model_best.pth.tar