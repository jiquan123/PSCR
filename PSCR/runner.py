# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from scipy import stats
from util import OPS, NOPS, RPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

#PSCR #RELATIVE SCORE
def net_forword_train_PSCR(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []

        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)

        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B//2, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)

            total_image = torch.cat([image_1, image_2], dim=0) #B, C, H, W
            B, C, H, W = total_image.shape
            if args.backbone == 'inceptionv4':
                start_idx = [0, 100, 213]
            else:
                start_idx = [0, 150, 288]
            #D = len(start_idx) ** 2
            #total_patch_images = OPS(total_image, args.image_size, start_idx) # B*D, C, 224, 224

            if args.PS_method == 'OPS':
                D = len(start_idx) ** 2
                total_patch_images = OPS(total_image, args.image_size, start_idx)
            elif args.PS_method == 'RPS':
                D = 9
                total_patch_images = RPS(total_image, args.image_size)
            else:
                D = 4
                total_patch_images = NOPS(total_image, args.image_size)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_patch_images)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            else:
                total_feature = backbone(total_patch_images).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            
            loss = criterion(preds_1, (label_1 - label_2).float().to(device)) + criterion(preds_2, (label_2 - label_1).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = label_2 + preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)

        return rho_s, rho_p


def net_forword_test_PSCR(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []

    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            total_image = torch.cat([image, target_img], dim=0)

            B, C, H, W = total_image.shape
            if args.backbone == 'inceptionv4':
                start_idx = [0, 100, 213]
            else:
                start_idx = [0, 150, 288]

            if args.PS_method == 'OPS':
                D = len(start_idx) ** 2
                total_patch_images = OPS(total_image, args.image_size, start_idx)
            elif args.PS_method == 'RPS':
                D = 9
                total_patch_images = RPS(total_image, args.image_size)
            else:
                D = 4
                total_patch_images = NOPS(total_image, args.image_size)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_patch_images)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)

            else:
                total_feature = backbone(total_patch_images).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
              
            pred = target_label + pred
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])

    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)

    return rho_s, rho_p

# PSCR ABSOLUTE SCORE
'''def net_forword_train(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []

        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)

        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B//2, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)

            total_image = torch.cat([image_1, image_2], dim=0) #B, C, H, W
            B, C, H, W = total_image.shape
            if args.backbone == 'inceptionv4':
                start_idx = [0, 100, 213]
            else:
                start_idx = [0, 150, 288]
            #D = len(start_idx) ** 2
            #total_patch_images = OPS(total_image, args.image_size, start_idx) # B*D, C, 224, 224

            if args.PS_method == 'OPS':
                D = len(start_idx) ** 2
                total_patch_images = OPS(total_image, args.image_size, start_idx)
            elif args.PS_method == 'RPS':
                D = 9
                total_patch_images = RPS(total_image, args.image_size)
            else:
                D = 4
                total_patch_images = NOPS(total_image, args.image_size)
            

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_patch_images)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            else:
                total_feature = backbone(total_patch_images).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            
            loss = criterion(preds_1, label_1.float().to(device)) + criterion(preds_2, label_2.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)

        return rho_s, rho_p


def net_forword_test(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []

    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            total_image = torch.cat([image, target_img], dim=0)

            B, C, H, W = total_image.shape
            if args.backbone == 'inceptionv4':
                start_idx = [0, 100, 213]
            else:
                start_idx = [0, 150, 288]
            
            if args.PS_method == 'OPS':
                D = len(start_idx) ** 2
                total_patch_images = OPS(total_image, args.image_size, start_idx)
            elif args.PS_method == 'RPS':
                D = 9
                total_patch_images = RPS(total_image, args.image_size)
            else:
                D = 4
                total_patch_images = NOPS(total_image, args.image_size)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_patch_images)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)

            else:
                total_feature = backbone(total_patch_images).reshape(D, B, -1).transpose(0, 1).mean(1)  # D*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
    
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])

    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)

    return rho_s, rho_p'''


#CR
def net_forword_train_CR(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []
        pred_1_scores = []
        pred_2_scores = []
        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)
        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)
            total_image = torch.cat([image_1, image_2], dim=0)
            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            pred_1_scores.extend([i.item() for i in preds_1])
            pred_2_scores.extend([i.item() for i in preds_2])
            loss = criterion(preds_1, (label_1 - label_2).float().to(device)) + criterion(preds_2, (label_2 - label_1).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = label_2 + preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)
        return rho_s, rho_p


def net_forword_test_CR(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []
    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)
    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        exemplar_image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
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
            pred = target_label + pred_1
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])
    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)
    return rho_s, rho_p


'''def net_forword_train(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []
        pred_1_scores = []
        pred_2_scores = []

        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)

        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)

            total_image = torch.cat([image_1, image_2], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)
            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                preds_2 = regressor(torch.cat([feature_2, feature_1], dim=-1)).view(-1)

            pred_1_scores.extend([i.item() for i in preds_1])
            pred_2_scores.extend([i.item() for i in preds_2])

            loss = criterion(preds_1, (label_1).float().to(device)) + criterion(preds_2, (label_2).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)

        return rho_s, rho_p


def net_forword_test(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []

    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        exemplar_image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            total_image = torch.cat([exemplar_image, target_img], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                

            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])

    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)

    return rho_s, rho_p'''


'''def net_forword_train(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []
        pred_1_scores = []

        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)

        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)

            total_image = torch.cat([image_1, image_2], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                
            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                

            pred_1_scores.extend([i.item() for i in preds_1])
            

            loss = criterion(preds_1, (label_1).float().to(device)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)

        return rho_s, rho_p


def net_forword_test(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []

    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        exemplar_image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            total_image = torch.cat([exemplar_image, target_img], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)

            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])

    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)

    return rho_s, rho_p'''

'''def net_forword_train(args, backbone, regressor, dataloaders, criterion, optimizer):
        true_scores = []
        pred_scores = []
        pred_1_scores = []
       

        backbone.train()
        regressor.train()
        torch.set_grad_enabled(True)

        for data, target in tqdm(dataloaders):
            true_scores.extend(data['label'].numpy())
            image_1 = data['img'].to(device)  # B, C, H, W
            label_1 = data['label'].to(device)
            image_2 = target['img'].to(device)
            label_2 = target['label'].to(device)

            total_image = torch.cat([image_1, image_2], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                
            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                preds_1 = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
                

            pred_1_scores.extend([i.item() for i in preds_1])
           

            loss = criterion(preds_1, (label_1 - label_2).float().to(device)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = label_2 + preds_1
            pred_scores.extend([i.item() for i in preds])
        rho_s, _ = stats.spearmanr(pred_scores, true_scores)
        rho_p, _ = stats.pearsonr(pred_scores, true_scores)

        return rho_s, rho_p


def net_forword_test(args, backbone, regressor, dataloaders):
    true_scores = []
    pred_scores = []

    backbone.eval()
    regressor.eval()
    torch.set_grad_enabled(False)

    for data, target_list in tqdm(dataloaders):
        true_scores.extend(data['label'].numpy())
        exemplar_image = data['img'].to(device)  # B, C, H, W
        target_img_list = [item['img'].float().cuda() for item in target_list]
        target_label_list = [item['label'].float().cuda() for item in target_list]
        preds = 0.0
        for target_img, target_label in zip(target_img_list, target_label_list):
            target_img = target_img.to(device)
            target_label = target_label.to(device)
            total_image = torch.cat([exemplar_image, target_img], dim=0)

            if args.backbone in ['vgg16', 'vgg19']:
                total_feature = backbone.features(total_image)
                total_feature = avgpool(total_feature).squeeze(2).squeeze(2)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)

            else:
                total_feature = backbone(total_image)  # 2*B, C
                feature_1 = total_feature[:total_feature.shape[0] // 2]
                feature_2 = total_feature[total_feature.shape[0] // 2:]
                pred = regressor(torch.cat([feature_1, feature_2], dim=-1)).view(-1)
            pred = target_label + pred
            preds += pred
        preds_avg = preds/len(target_label_list)
        pred_scores.extend([i.item() for i in preds_avg])

    rho_s, _ = stats.spearmanr(pred_scores, true_scores)
    rho_p, _ = stats.pearsonr(pred_scores, true_scores)

    return rho_s, rho_p'''

