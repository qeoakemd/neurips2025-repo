import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from util_param_ import *

import torchvision
import torchvision.transforms as transfroms


from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)

# from torchvision.datasets import CocoDetection
# weights   = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
# weights   = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
# transform = weights.transforms()  # 

train_set = CocoDetectionDataset(
    root='./data/COCO/train2017',
    annFile='./data/COCO/annotations/instances_train2017.json',
    # transforms=transform   
    alb_transforms=transform   
)
test_set = CocoDetectionDataset(
    root='./data/COCO/val2017',
    annFile='./data/COCO/annotations/instances_val2017.json',
    # transforms=transform   
    alb_transforms=transform   
)

train_set = Subset(train_set, list(range(m)))

indices = list(range(len(train_set)))
random.shuffle(indices)
partitions = np.array_split(indices, n)
train_subsets = [Subset(train_set, p.tolist()) for p in partitions]

test_set = Subset(test_set, list(range(test_batch_size)))

indices = list(range(len(test_set)))
random.shuffle(indices)
partitions = np.array_split(indices, n)
test_subsets = [Subset(test_set, p.tolist()) for p in partitions]


partitions = np.array_split(range(m), n)

# beta_gd = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_sgc = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_proposed0 = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_bgc = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_ehd = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_od = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)
# beta_issgd = fasterrcnn_resnet50_fpn(weights=None, num_classes=91).to(device)

beta_gd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_sgc = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_proposed0 = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_bgc = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_ehd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_od = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_issgd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)


initial_state_dict = beta_gd.state_dict()
beta_sgc.load_state_dict(initial_state_dict)
beta_proposed0.load_state_dict(initial_state_dict)
beta_bgc.load_state_dict(initial_state_dict)
beta_ehd.load_state_dict(initial_state_dict)
beta_od.load_state_dict(initial_state_dict)
beta_issgd.load_state_dict(initial_state_dict)

iteration = 0
losses_gd = []
losses_proposed0 = []
losses_sgc = []
losses_bgc = []
losses_ehd = []
losses_od = []
losses_issgd = []
itq = 0
for t in range(T):
    stragglers = np.random.rand(n) < p  # Determine stragglers for this iteration
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
    it2 = 0
    for cid, subset in enumerate(train_subsets):
        loader = DataLoader(
            subset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
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

        for images, targets in loader:
            X_batch = [img.to(device) for img in images]
            y_batch = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
            
            for i in range(len(grad_gd_tmp)):
                gradient_gd_sum[i] += grad_gd_tmp[i]
                gradient_proposed0_sum[i] += grad_proposed0_tmp[i] 
                gradient_sgc_sum[i] += grad_sgc_tmp[i]
                gradient_bgc_sum[i] += grad_bgc_tmp[i]
                gradient_ehd_sum[i] += grad_ehd_tmp[i]
                gradient_od_sum[i] += grad_od_tmp[i]
                gradient_issgd_sum[i] += grad_issgd_tmp[i]
            it2 += 1
        it += 1
        
    
    for i in range(len(gradient_gd_sum)):
        grad_gd.append(gradient_gd_sum[i] / it)
        grad_proposed0.append(gradient_proposed0_sum[i] / it)
        grad_sgc.append(gradient_sgc_sum[i] / it)
        grad_bgc.append(gradient_bgc_sum[i] / it)
        grad_ehd.append(gradient_ehd_sum[i] / it)
        grad_od.append(gradient_od_sum[i] / it)
        grad_issgd.append(gradient_issgd_sum[i] / it)

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

            loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn
            )

            for images, targets in loader:
                X_test = [img.to(device) for img in images]
                y_test = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                i += len(X_test)
                
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

            print(f"iteration {t}, Validation Loss: {loss_proposed0 / i:.4f}, (SGC: {loss_sgc / i:.4f}, GD: {loss_gd / i:.4f}, BGC: {loss_bgc / i:.4f}, EHD: {loss_ehd / i:.4f}, OD: {loss_od / i:.4f}, IS-SGD: {loss_issgd / i:.4f})")


            losses_gd.append(loss_gd / i)
            losses_proposed0.append(loss_proposed0 / i)
            losses_sgc.append(loss_sgc / i)
            losses_bgc.append(loss_bgc / i)
            losses_ehd.append(loss_ehd / i)
            losses_od.append(loss_od / i)
            losses_issgd.append(loss_issgd / i)
        

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