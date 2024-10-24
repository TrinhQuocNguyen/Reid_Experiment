# python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                        --logs-dir logs/cuhk03np2market/target_fine_tuning_900_global_Fuse_ECAB \
#                        --initial-weights logs/cuhk03np2market/source_pretraining -b 128 \
#                        --num-clusters 900 --arch resnet101

# python model_test.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                     --resume logs/cuhk03np2market/target_fine_tuning_700_global_Fuse_ECAB/model_best.pth.tar \
#                     --num-classes 700 --arch resnet101


# NOTE STARTGAN
# python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                        --logs-dir logs/duke2market_stargan/target_fine_tuning_900_global_Fuse_ECAB_starGAN \
#                        --initial-weights /mnt/AIProjects/trinh/Projects/DAPRH/saves/reid/duke2market/S1/R50Mix -b 128 \
#                        --num-clusters 900 --arch resnet101


## step 1 Source-domain pre-training
# python source_pretrain.py -ds cuhk03np -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
#                           --logs-dir logs/cuhk03np2market/resnet34/source_pretraining -b 128 --arch resnet34_source \
#                           --epochs 2 --iters 2 --eval-step 1

## step 2 Target-domain fine-tuning                        
python target_train.py -dt market --data-dir /mnt/AIProjects/trinh/DATA/reid \
                       --logs-dir logs/cuhk03np2market/resnet34/target_fine_tuning_900_global_FUSE \
                       --initial-weights logs/cuhk03np2market/resnet34/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet34 \
                       --epochs 2 --iters 2 --eval-step 1
