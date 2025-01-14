## step 1 Source-domain pre-training
# Market to Duke
# CUDA_VISIBLE_DEVICES=1,2 python source_pretrain.py -ds market -dt duke --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2duke_101/source_pretraining -b 128 --arch resnet101_source

## step 2 Target-domain fine-tuning
# Market to Duke
# CUDA_VISIBLE_DEVICES=1,2 python target_train.py -dt duke --data-dir /old/home/ccvn/Workspace/trinh/data/reid --logs-dir logs/market2duke_101/target_fine_tuning --initial-weights logs/market2duke_101/source_pretraining -b 128 --arch resnet101

## step 3 Evaluate in the target domain
# Market to Duke
CUDA_VISIBLE_DEVICES=1,2 python model_test.py -dt duke --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/market2duke_101/target_fine_tuning/model_best.pth.tar --arch resnet101