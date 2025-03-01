import torch
import torch.nn as nn
import os
import sys
from backbone.resnet import resnet18_backbone, resnet50_backbone
from backbone.inceptionv4 import inceptionv4
from backbone.vgg import vgg16, vgg19
from backbone.vit import ViTExtractor, SwinExtractor, MAE
from dataloader.AGIQA1K import get_AGIQA1K_dataloaders
from dataloader.AGIQA3K import get_AGIQA3Kq_dataloaders, get_AGIQA3Kc_dataloaders
from dataloader.AIGCIQA2023 import get_AIGCIQA2023q_dataloaders, get_AIGCIQA2023a_dataloaders, get_AIGCIQA2023c_dataloaders
from regressor import MLP
from runner import net_forword_train_PSCR, net_forword_test_PSCR, net_forword_train_CR, net_forword_test_CR
from config import get_parser
from util import get_logger, log_and_print

import random
import torch.backends.cudnn as cudnn

sys.path.append('../')
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    base_logger = get_logger(f'exp/PSCR.log', args.log_info)

    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        #print(f'{k}: {v}')
        log_and_print(base_logger, f'{k}: {v}')
    print('=' * 40)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if args.backbone == 'resnet50':
        backbone = resnet50_backbone().to(device)
        regressor = MLP(4096).to(device)
    elif args.backbone == 'resnet18':
        backbone = resnet18_backbone().to(device)
        regressor = MLP(1024).to(device)
    elif args.backbone == 'vgg16':
        backbone = vgg16().to(device)
        regressor = MLP(1024).to(device)
    elif args.backbone == 'vgg19':
        backbone = vgg19().to(device)
        regressor = MLP(1024).to(device)
    elif args.backbone == 'vit':
        backbone = ViTExtractor().to(device)
        regressor = MLP(2048).to(device)
    elif args.backbone == 'swin':
        backbone = SwinExtractor().to(device)
        regressor = MLP(2048).to(device)
    elif args.backbone == 'mae':
        backbone = MAE().to(device)
        regressor = MLP(1536).to(device)
    else:
        backbone = inceptionv4(num_classes=1000, pretrained='imagenet').to(device)
        regressor = MLP(3072).to(device)

    if args.benchmark == 'AIGCIQA2023q':
        dataloaders = get_AIGCIQA2023q_dataloaders(args)
    elif args.benchmark == 'AIGCIQA2023a':
        dataloaders = get_AIGCIQA2023a_dataloaders(args)
    elif args.benchmark == 'AIGCIQA2023c':
        dataloaders = get_AIGCIQA2023c_dataloaders(args)
    elif args.benchmark == 'AGIQA3Kq':
        dataloaders = get_AGIQA3Kq_dataloaders(args)
    elif args.benchmark == 'AGIQA3Kc':
        dataloaders = get_AGIQA3Kc_dataloaders(args)
    else:
        dataloaders = get_AGIQA1K_dataloaders(args)

    criterion = nn.MSELoss(reduction='mean').cuda()
    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam([*backbone.parameters()] + [*regressor.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_s_best = 0.0
    rho_p_best = 0.0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')
        if args.PS: 
            rho_s, rho_p = net_forword_train_PSCR(args, backbone, regressor, dataloaders['train'], criterion, optimizer)
            log_and_print(base_logger, f'train spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')
            rho_s, rho_p = net_forword_test_PSCR(args, backbone, regressor, dataloaders['test'])
            log_and_print(base_logger, f'test spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')
        else:
            rho_s, rho_p = net_forword_train_CR(args, backbone, regressor, dataloaders['train'], criterion, optimizer)
            log_and_print(base_logger, f'train spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')
            rho_s, rho_p = net_forword_test_CR(args, backbone, regressor, dataloaders['test'])
            log_and_print(base_logger, f'test spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')
 
        if rho_s > rho_s_best:
            rho_s_best = rho_s
            epoch_best = epoch
            log_and_print(base_logger, '##### New best spearmanr correlation #####')
            path = 'ckpts/' + 'best_model.pt'
            torch.save({'epoch': epoch,
                        'backbone': backbone.state_dict(),
                        'regressor': regressor.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_s_best}, path)
        if rho_p > rho_p_best:
            rho_p_best = rho_p
            log_and_print(base_logger, '##### New best pearsonr correlation #####')
        log_and_print(base_logger, ' EPOCH_best: %d, SRCC_best: %.6f, PLCC_best: %.6f' % (epoch_best, rho_s_best, rho_p_best))



