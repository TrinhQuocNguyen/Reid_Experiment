
## step 2 Target-domain fine-tuning                        
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid/Market1501 \
                       --logs-dir logs/cuhk03np2market/resnet101/ECBAX_SABX \
                       --initial-weights /old/home/ccvn/Workspace/trinh/CORE_ReID/logs/cuhk03np2market/resnet101/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet101 
