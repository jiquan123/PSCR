import torch
import torch.nn as nn
import os
import sys
from scipy import stats
from tqdm import tqdm
from backbone.resnet import resnet18_backbone, resnet50_backbone
from backbone.inceptionv4 import inceptionv4
from backbone.vgg import vgg16, vgg19
from dataloader.AGIQA1K import get_AGIQA1K_dataloaders
from config import get_parser
from regressor import MLP
from util import get_logger, log_and_print

import random
import torch.backends.cudnn as cudnn

sys.path.append('../')
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    base_logger = get_logger(f'exp/CASE.log', args.log_info)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.backbone == 'resnet50':
        backbone = resnet50_backbone().to(device)
        regressor = MLP(2048 * 2).to(device)
    elif args.backbone == 'resnet18':
        backbone = resnet18_backbone().to(device)
        regressor = MLP(512 * 2).to(device)
    elif args.backbone == 'vgg16':
        backbone = vgg16().to(device)
        regressor = MLP(512 * 2).to(device)
    elif args.backbone == 'vgg19':
        backbone = vgg19().to(device)
        regressor = MLP(512 * 2).to(device)
    else:
        backbone = inceptionv4(num_classes=1000, pretrained='imagenet').to(device)
        regressor = MLP(1536 * 2).to(device)

    
    dataloaders = get_AGIQA1K_dataloaders(args)
    checkpoint = torch.load('./ckpts/best_model.pt')
        

# 更新模型权重
    backbone.load_state_dict(checkpoint['backbone'])
    regressor.load_state_dict(checkpoint['regressor'])
    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    true_scores = []
    pred_scores = []

    for data, target_list in tqdm(dataloaders['test']):
        true_scores.extend(data['label'].numpy())
        log_and_print(base_logger, f'Query_Ground-truth: {true_scores}')
        exemplar_image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            Delta = data['label'].cuda()-target_label
            log_and_print(base_logger, f'Exemplar_Ground-truth: {target_label}')
            log_and_print(base_logger, f'Delta: {Delta}')
            total_image = torch.cat([exemplar_image, target_img], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
            log_and_print(base_logger, f'Pred_delta: {pred_1}')
            pred = target_label + pred_1
            preds += pred
        preds_avg = preds/len(target_label_list)
        log_and_print(base_logger, f'Pred_avg: {preds_avg}')
        pred_scores.extend([i.item() for i in preds_avg])

                


               



                
                

                

            

        

