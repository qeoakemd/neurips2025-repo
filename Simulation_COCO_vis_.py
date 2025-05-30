import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from util_ import *

import torchvision
import torchvision.transforms as transfroms
from torchvision.datasets import CocoDetection

import albumentations as AA
from albumentations.pytorch import ToTensorV2

from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    fasterrcnn_mobilenet_v3_large_fpn,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


batch_size = test_batch_size  
coco_category_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

category_id_to_index = {id: idx for idx, id in enumerate(coco_category_ids)}

##############################################################################################################################
# Load COCO dataset

def collate_fn(batch):
    return tuple(zip(*batch))

##############################################################################################################################
##############################################################################################################################

# partitions = np.array_split(range(m), n)

#############################################################################################################################################################
''' Data loader'''
#############################################################################################################################################################
# weights   = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

# beta_gd =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_sgc =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_proposed0 =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_bgc =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_ehd =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_od =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)
# beta_issgd =  fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91).to(device)


weights   = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1

beta_gd = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_sgc = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_proposed0 = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_bgc = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_ehd = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_od = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)
beta_issgd = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91).to(device)



transform = weights.transforms()  
dataset = CocoDetection(
    root='./data/COCO/val2017',
    annFile='./data/COCO/annotations/instances_val2017.json',
    transform=transform
)

data_loader = DataLoader(
                    Subset(dataset, list(range(2*n))),
                        batch_size=2,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn)

initial_state_dict = beta_gd.state_dict()
beta_sgc.load_state_dict(initial_state_dict)
beta_proposed0.load_state_dict(initial_state_dict)
beta_bgc.load_state_dict(initial_state_dict)
beta_ehd.load_state_dict(initial_state_dict)
beta_od.load_state_dict(initial_state_dict)
beta_issgd.load_state_dict(initial_state_dict)
##############################################################################################################################

# Early stopping variables
best_loss = float('inf')
stopping_step = 0
T = 1000 
gamma_t = lambda t: 0.1

for t in range(T):
    stragglers = np.random.rand(n) < p  
    count = 0

    grad_gd = []
    grad_proposed0 = []
    grad_sgc = []
    grad_bgc = []
    grad_ehd = []
    grad_od = []
    grad_issgd = []

    gradient_gd_sum = [torch.zeros_like(param).to(device) for param in beta_gd.parameters()]
    gradient_proposed0_sum = [torch.zeros_like(param).to(device) for param in beta_proposed0.parameters()]
    gradient_sgc_sum = [torch.zeros_like(param).to(device) for param in beta_sgc.parameters()]
    gradient_bgc_sum = [torch.zeros_like(param).to(device) for param in beta_bgc.parameters()]
    gradient_ehd_sum = [torch.zeros_like(param).to(device) for param in beta_ehd.parameters()]
    gradient_od_sum = [torch.zeros_like(param).to(device) for param in beta_od.parameters()]
    gradient_issgd_sum = [torch.zeros_like(param).to(device) for param in beta_issgd.parameters()]
    it = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        formatted_targets = []
        for anns in targets:
            if len(anns) > 0:
                boxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
                boxes[:, 2:] += boxes[:, :2]  # [x,y,w,h] → [x1,y1,x2,y2]
                labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
            else:
                boxes  = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,),    dtype=torch.int64)

            formatted_targets.append({
                'boxes':  boxes.to(device),
                'labels': labels.to(device),
            })
        X_batch = images
        y_batch = formatted_targets

        stragglers_tmp_proposed0 = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_sgc = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_bgc = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_od = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_ehd = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_issgd = np.ones_like(stragglers, dtype=bool)

        stragglers_tmp_proposed0[np.where(A0[:,it] != 0)[0]] = stragglers[np.where(A0[:,it] != 0)[0]]
        stragglers_tmp_sgc[np.where(A2[:,it] != 0)[0]] = stragglers[np.where(A2[:,it] != 0)[0]]
        stragglers_tmp_bgc[np.where(A3[:,it] != 0)[0]] = stragglers[np.where(A3[:,it] != 0)[0]]
        stragglers_tmp_od[np.where(A6[:,it] != 0)[0]] = stragglers[np.where(A6[:,it] != 0)[0]]
        stragglers_tmp_ehd[np.where(A4[:,it] != 0)[0]] = stragglers[np.where(A4[:,it] != 0)[0]]
        stragglers_tmp_issgd[np.where(A5[:,it] != 0)[0]] = stragglers[np.where(A5[:,it] != 0)[0]]

        

        grad_gd_tmp = gradient_GD_loader(X_batch, beta_gd, y_batch) 
        beta_gd.zero_grad()

        grad_proposed0_tmp = gradient_GC_loader(X_batch, beta_proposed0, y_batch, stragglers_tmp_proposed0, it, A0, w0) 
        beta_proposed0.zero_grad()

        grad_sgc_tmp = gradient_GC_loader(X_batch, beta_sgc, y_batch, stragglers_tmp_sgc, it, A2, np.ones(n)) 
        beta_sgc.zero_grad()
        
        grad_bgc_tmp = gradient_GC_loader(X_batch, beta_bgc, y_batch, stragglers_tmp_bgc, it, A3, np.ones(n)) 
        beta_bgc.zero_grad()
        
        grad_ehd_tmp = gradient_GC_loader(X_batch, beta_ehd, y_batch, stragglers_tmp_ehd, it, A4, np.ones(n))
        beta_ehd.zero_grad()
        
        w_od = calculate_optimal_decoding(A6, stragglers)
        grad_od_tmp = gradient_GC_loader(X_batch, beta_od, y_batch, stragglers_tmp_od, it, A6, w_od) 
        beta_od.zero_grad()
        
        grad_issgd_tmp = gradient_GC_loader(X_batch, beta_issgd, y_batch, stragglers_tmp_issgd, it, A5, np.ones(n))
        beta_issgd.zero_grad()

        it += 1
        
        for i in range(len(grad_gd_tmp)):
            gradient_gd_sum[i] += grad_gd_tmp[i]
            gradient_proposed0_sum[i] += grad_proposed0_tmp[i] 
            gradient_sgc_sum[i] += grad_sgc_tmp[i]
            gradient_bgc_sum[i] += grad_bgc_tmp[i]
            gradient_ehd_sum[i] += grad_ehd_tmp[i]
            gradient_od_sum[i] += grad_od_tmp[i]
            gradient_issgd_sum[i] += grad_issgd_tmp[i]
        

    for i in range(len(gradient_gd_sum)):
        grad_gd.append(gradient_gd_sum[i] / it) 
        grad_proposed0.append(gradient_proposed0_sum[i] / it) 
        grad_sgc.append(gradient_sgc_sum[i] / it) 
        grad_bgc.append(gradient_bgc_sum[i] / it)
        grad_ehd.append(gradient_ehd_sum[i] / it)  
        grad_od.append(gradient_od_sum[i] / it)  
        grad_issgd.append(gradient_issgd_sum[i] / it)
    # Validation and early stopping check
    if t % period_validate == 0:
        with torch.no_grad():
            i = 0
            
            loss_gd = 0
            loss_sgd = 0
            loss_proposed0 = 0
            loss_proposed = 0
            loss_sgc = 0
            loss_bgc = 0
            loss_ehd = 0
            loss_od = 0
            loss_issgd = 0


            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                formatted_targets = []
                for anns in targets:
                    if len(anns) > 0:
                        boxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
                        boxes[:, 2:] += boxes[:, :2]  # [x,y,w,h] → [x1,y1,x2,y2]
                        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
                    else:
                        boxes  = torch.zeros((0, 4), dtype=torch.float32)
                        labels = torch.zeros((0,),    dtype=torch.int64)

                    formatted_targets.append({
                        'boxes':  boxes.to(device),
                        'labels': labels.to(device),
                    })
                X_test = images
                y_test = formatted_targets
                
                i += batch_size
                

                fig_plot(beta_gd, 'gd')
                fig_plot(beta_proposed0, 'proposed0')
                fig_plot(beta_sgc, 'sgc')
                fig_plot(beta_bgc, 'bgc')
                fig_plot(beta_ehd, 'ehd')
                fig_plot(beta_od, 'od')
                fig_plot(beta_issgd, 'issgd')

                beta_gd.train()
                beta_sgc.train()
                beta_proposed0.train()
                beta_bgc.train()
                beta_ehd.train()
                beta_od.train()
                beta_issgd.train()

                loss_gd += sum(loss for loss in beta_gd(X_test, y_test).values()).item()  
                loss_sgc += sum(loss for loss in beta_sgc(X_test, y_test).values()).item()  
                loss_proposed0 += sum(loss for loss in beta_proposed0(X_test, y_test).values()).item()  
                loss_bgc += sum(loss for loss in beta_bgc(X_test, y_test).values()).item()  
                loss_ehd += sum(loss for loss in beta_ehd(X_test, y_test).values()).item() 
                loss_od += sum(loss for loss in beta_od(X_test, y_test).values()).item()  
                loss_issgd += sum(loss for loss in beta_issgd(X_test, y_test).values()).item() 

            print(f"iteration {t}, Validation Loss: {loss_proposed0:.4f} (SGC: {loss_sgc:.4f}, GD: {loss_gd:.4f}, BGC: {loss_bgc:.4f}, EHD: {loss_ehd:.4f}, OD: {loss_od:.4f}, IS-SGD: {loss_issgd:.4f})")


    ###############################################################################################################
    '''
    Update model parameters
    '''
    ###############################################################################################################
    with torch.no_grad():
        count_tmp = 0
        for param in beta_gd.parameters():
            param -= gamma_t(t) * grad_gd[count_tmp] / norm_gd
            count_tmp += 1
        count_tmp = 0
        for param in beta_sgc.parameters():
            param -= gamma_t(t) * grad_sgc[count_tmp] / norm_sgc
            count_tmp += 1
        count_tmp = 0
        for param in beta_proposed0.parameters():
            param -= gamma_t(t) * grad_proposed0[count_tmp] / norm_proposed
            count_tmp += 1
        count_tmp = 0
        for param in beta_bgc.parameters():
            param -= gamma_t(t) * grad_bgc[count_tmp] / norm_bgc
            count_tmp += 1
        count_tmp = 0
        for param in beta_ehd.parameters():
            param -= gamma_t(t) * grad_ehd[count_tmp] / norm_ehd
            count_tmp += 1
        count_tmp = 0
        for param in beta_od.parameters():
            param -= gamma_t(t) * grad_od[count_tmp] / norm_od
            count_tmp += 1
        count_tmp = 0
        for param in beta_issgd.parameters():
            param -= gamma_t(t) * grad_issgd[count_tmp] / norm_issgd
            count_tmp += 1
        
