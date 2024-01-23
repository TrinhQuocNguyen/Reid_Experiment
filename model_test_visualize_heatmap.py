from __future__ import print_function, absolute_import
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

import warnings
warnings.filterwarnings('ignore')

import argparse
import os.path as osp
import random
import numpy as np

import sys
sys.path.append('..')

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint
from reid.models.resnet import Encoder


import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib.pyplot import imshow
from torch import topk

device_ids = [0,1]

def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, test_loader


def create_model(args, extract_feat_):  
    arch = args.arch
    model_student = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_classes,\
                                  num_split=args.split_parts, extract_feat=extract_feat_).cuda()
    model_teacher = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_classes,\
                                  num_split=args.split_parts, extract_feat=extract_feat_).cuda()
    model_student = nn.DataParallel(model_student)  
    model_teacher = nn.DataParallel(model_teacher)

    for param in model_teacher.parameters():
        param.detach_()

    return model_student, model_teacher


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
 
def main_worker(args):
    
    ### Load data here
    image = Image.open("0001_c1s1_001051_00.jpg")
    
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])
    
    display_transform = transforms.Compose([
    transforms.Resize((224,224))])
    tensor = preprocess(image)
    prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
    
    ### Load model here
    # Create model
    model_student, model_teacher = create_model(args, False)
    # print("MODEL STUDENT:")
    # print(model_student)
    # print("MODEL TEACHER:")
    # print(model_teacher)
    encoder = Encoder(model_student, model_teacher)
    
    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    encoder.load_state_dict(checkpoint['state_dict'])
    # model_teacher.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))   
    
    final_layer = model._modules.get('layer4')

    activated_features = SaveFeatures(final_layer)
    prediction = encoder.model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    print(topk(pred_probabilities,1))
    
    
    weight_softmax_params = list(encoder.model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params
    # class_idx = topk(pred_probabilities,1)[1].int()
    class_idx = class_idx = topk(pred_probabilities,1)[1].int()
    overlay = getCAM(activated_features.features, weight_softmax, class_idx )
    plt.savefig('foo.png', overlay[0], alpha=0.5, cmap='jet')
    plt.savefig('2.png', display_transform(image))
    plt.savefig('3.png', skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
        
    
    
    
    # cudnn.benchmark = True

    # log_dir = osp.dirname(args.resume)
    # sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    # # Create data loaders
    # dataset_target, test_loader_target = \
    #     get_data(args.dataset_target, args.data_dir, args.height,
    #              args.width, args.batch_size, args.workers)

    # # Create model
    # model_student, model_teacher = create_model(args, False)
    # print("MODEL STUDENT:")
    # print(model_student)
    # print("MODEL TEACHER:")
    # print(model_teacher)
    # encoder = Encoder(model_student, model_teacher)
    
    # # Load from checkpoint
    # checkpoint = load_checkpoint(args.resume)
    # encoder.load_state_dict(checkpoint['state_dict'])
    # start_epoch = checkpoint['epoch']
    # best_mAP = checkpoint['best_mAP']
    # print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
    
    

    # # # Evaluator
    # print("Test on the target domain of {}:".format(args.dataset_target))
    # evaluator_ = Evaluator(encoder)
    # evaluator_.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, source=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str,  default='duke',choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str,  default='resnet50')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--split-parts', type=int, default=2)       # splitted parts
    parser.add_argument('--num-classes', type=int, default=700)       # cluster classes
    # testing configs
    parser.add_argument('--resume', type=str, metavar='PATH',\
                         default='logs_m2d_clusters_900/model_best.pth.tar')
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/dj/reid/data')
    main()
