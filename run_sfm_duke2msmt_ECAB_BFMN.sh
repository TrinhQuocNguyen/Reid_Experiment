# Market to Duke
CUDA_VISIBLE_DEVICES=2,3 python model_save_feature_maps.py -dt msmt17 --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/duke2msmt_ECAB_BFMN/target_fine_tuning_2500/model_best.pth.tar --num-classes 2500 --arch resnet101 --batch-size 4