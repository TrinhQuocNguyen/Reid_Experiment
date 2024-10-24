## step 2 Target-domain fine-tuning
# for example, duke-to-market
python target_train.py -dt market \
        --data-dir /home/ccvn/Workspace/trinh/data/reid \
        --logs-dir logs/duke2market_stargan/target_fine_tuning_101 \
        --initial-weights /mnt/AIProjects/trinh/Projects/DAPRH/saves/reid/duke2market/S1/R50Mix \
        -b 64 --num-clusters 700 --arch resnet101

## step 3 Evaluate in the target domain
# for example, duke-to-market
# python model_test.py -dt market \
#         --data-dir /home/ccvn/Workspace/trinh/data/reid \
#         --resume /mnt/AIProjects/trinh/Projects/DAPRH/pretrained/duke2market.pth.tar \
#         --num-classes 700 --arch resnet50