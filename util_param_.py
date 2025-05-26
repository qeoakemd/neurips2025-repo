import numpy as np
import random
import torch

from datetime import datetime
import copy
import csv
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

now = datetime.now()
torch.cuda.empty_cache()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.datasets import CocoDetection

import albumentations as AA
from albumentations.pytorch import ToTensorV2

transform = AA.Compose(
    [
        AA.RandomSizedBBoxSafeCrop(width=480, height=480, erosion_rate=0.0, p=1.0),  
        # AA.RandomSizedBBoxSafeCrop(width=320, height=320, erosion_rate=0.0, p=1.0),  
        AA.HorizontalFlip(p=0.5),
        AA.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        AA.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=AA.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.05,          #
    ),
)
class CocoDetectionDataset(CocoDetection):
    def __init__(self, root, annFile, alb_transforms=None):
        # base transforms=None to avoid base applying Albumentations
        super().__init__(root, annFile, transforms=None, transform=None, target_transform=None)
        self.alb_transforms = alb_transforms
        self.valid_indices = [i for i, img_id in enumerate(self.ids)
                              if len(self.coco.imgToAnns[img_id]) > 0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        # super returns (PIL.Image, annotations_list)
        img, annots = super().__getitem__(actual_idx)

        bboxes = []
        labels = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            # bboxes.append([x, y, x + w, y + h])
            # labels.append(ann['category_id'])
            
            x1, y1, x2, y2 = x, y, x + w, y + h
            if (x2 > x1) and (y2 > y1):
                bboxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if len(bboxes) == 0:
            return None
        
        sample = self.alb_transforms(image=np.array(img), bboxes=bboxes, labels=labels)
        image = sample['image']               # Tensor[3,H,W]
        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([self.ids[actual_idx]]),
        }
        return image, target

# collate_fn: list of (image, target) → (list(images), list(targets))
def collate_fn(batch):
    # images, targets = list(zip(*batch))
    # return list(images), list(targets)
    images, targets = [], []
    for img, tgt in batch:
        boxes = tgt['boxes']
        if boxes.ndim == 1:
            boxes = boxes.view(-1, 4)  # ⇒ (0,4)

        images.append(img)
        targets.append({**tgt, 'boxes': boxes})
    return images, targets


def gradient_GD_loader(X, model, y):
    gradient_sum = [torch.zeros_like(param).to(device) for param in model.parameters()]

    model.train()
    loss_dict = model(X, y)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()

    # Compute gradients and update gradient_sum
    count_temp = 0
    for param in model.parameters():
        if param.grad is not None:
            gradient_sum[count_temp] += clip_grad(param.grad.clone(), C)
        count_temp += 1

    return gradient_sum

def gradient_GC_loader(X, model, y, stragglers, data_idx, encoding_mat, decoding_vec):
    gradient_sum = [torch.zeros_like(param).to(device) for param in model.parameters()]
    idx_nonstragg = np.where(stragglers == False)[0]
    for worker_id in idx_nonstragg:
        decoding_weight = decoding_vec[worker_id]
        encoding_weight = encoding_mat[worker_id,data_idx]
        model.zero_grad()
        
        model.train()
        loss_dict = model(X, y)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        count_temp = 0
        for param in model.parameters():
            if param.grad is not None:
                gradient_sum[count_temp] += param.grad.clone() * encoding_weight * decoding_weight
            count_temp += 1
            

    for i in range(len(gradient_sum)):
        gradient_sum[i] = clip_grad(gradient_sum[i], C)
        
    return gradient_sum

def genran_vec(n, d_avg):
    d_i = [random.randint(mn, mx) for _ in range(n)]
    while (sum(d_i) / n) >= d_avg + 0.05 or (sum(d_i) / n) <= d_avg - d_gap:
        d_i = [random.randint(mn, mx) for _ in range(n)]
    return d_i

def clip_grad(grad, C):
    return torch.clip(grad, -C, C)

def calculate_optimal_decoding(assignment_matrix, stragglers):
    n, m = assignment_matrix.shape
    A = assignment_matrix.copy()

    if isinstance(stragglers, np.ndarray) and stragglers.dtype == bool:
        if len(stragglers) != m:
            raise ValueError("Stragglers array length must match the number of machines.")
        A[:, stragglers] = 0
    else:
        for s in stragglers:
            A[:, s] = 0  

    try:
        pseudo_inverse = np.linalg.pinv(A)
    except np.linalg.LinAlgError as e:
        print("Error in pseudoinverse calculation:", e)
        return None
    
    ones_vector = np.ones(n)
    optimal_decoding = pseudo_inverse @ ones_vector

    return optimal_decoding

m = 100000 
test_batch_size = 5000
batch_size = 32
T = 101
gamma = 0.01
gamma_t = lambda t: gamma # 
n = 10 
Z_th = 1.1 
mu_min = 0.01
mu_max = 2 
mu = np.random.uniform(mu_min, mu_max, n)
p = np.exp(- mu * (Z_th - 1)) 
p_avg = np.mean(p)
sigma = 0.2 
alpha_param = 2 
period_validate = 10 
period_disp = 10
period_straggler = 1
n_split = n 
d = 2
d2 = 2 
d_gap = 0.1
C = np.inf

partitions = np.array_split(range(m), n_split)

b = np.zeros(n, dtype=int)
b[-1] = 1
b[:-1] = 2
b[0] = b[0] + n + n_split - 2 - sum(b[:-1])

delta_inv = (1-p) / p
sum_delta_inv = sum(delta_inv)
Y1 = delta_inv * n / sum_delta_inv
alpha = np.zeros((n,n_split))


avg_num = 1
avg_num2 = ((n + n_split - 1) / n) 
avg_num3 = d

Y = delta_inv * n_split / sum_delta_inv
prev_idx = 0
for i in range(n): 
    if i == 0:
        alpha[i, prev_idx] = 1
    else:
        alpha[i, prev_idx] = 1 - alpha[i - 1, prev_idx]
    alpha[i, prev_idx + 1: prev_idx + b[i] - 1] = np.ones_like(alpha[i, prev_idx + 1: prev_idx + b[i] - 1])
    alpha[i, prev_idx + b[i] - 1] = Y[i] - sum(alpha[i, :prev_idx + b[i] - 1])

    prev_idx = prev_idx + b[i] - 1

alpha0 = np.zeros((n,n_split))
mn = 0
mx = np.ceil(2 * d)
prev_idx = 0
for i in range(n): 
    alpha0[i, 0] = Y[i] - (b[i] - 1)
    alpha0[i, prev_idx + 1: prev_idx + b[i]] = np.ones_like(alpha0[i, prev_idx + 1: prev_idx + b[i]])

    prev_idx = prev_idx + b[i] - 1

w_tilde0 = np.random.randn(n) 

w0 = w_tilde0 / (1-p)
A0 = np.zeros_like(alpha0)
for i in range(n):
    for j in range(n_split):
        A0[i,j] = alpha0[i,j] / w_tilde0[i]

workers_proposed = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(alpha[i,:] != 0)[0]
    for data in assigned_workers:
        workers_proposed[i].append(data)

d_i = np.zeros(n_split)
d_i = genran_vec(n_split, d)


A2 = np.zeros_like(alpha)
workers_sgc = [[] for _ in range(n)]
for i in range(n_split): 
    assigned_workers = np.random.choice(n, size = d_i[i], replace=False)
    for worker in assigned_workers:
        workers_sgc[worker].append(i) 
        A2[worker, i] = 1 / ((1-p[worker])*d_i[i])
            
def bernoulli(p):
    return 1 if np.random.rand() < p else 0
A3 = np.zeros_like(alpha)
for i in range(n):
    for j in range(n_split):
        A3[i,j] = bernoulli(d2/n)

workers_bgc = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A3[i,:] != 0)[0]
    for data in assigned_workers:
        workers_bgc[i].append(data)
        
A4 = np.zeros_like(alpha)
sp = 0
prev = 0 
for i in range(n):
    if np.floor(i/d2) != prev:
        sp += int(d2 * n_split / n)
        prev = np.floor(i/d2)
    A4[i,sp:int(sp+d2 * n_split / n)] = 1


workers_erasurehead = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A4[i,:] != 0)[0]
    for data in assigned_workers:
        workers_erasurehead[i].append(data)

A5 = np.zeros_like(alpha)
sp = 0
prev = 0 
for i in range(n):
    A5[i,sp:int(sp+n_split / n)] = 1
    sp += int(n_split / n)
    prev = np.floor(i/d2)


workers_issgd = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A5[i,:] != 0)[0]
    for data in assigned_workers:
        workers_issgd[i].append(data)

G = nx.random_regular_graph(d2, n_split, seed=42)
A6 = nx.adjacency_matrix(G).toarray()

workers_od = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A6[i,:] != 0)[0]
    for data in assigned_workers:
        workers_od[i].append(data)


norm_gd = avg_num 
norm_proposed = avg_num2 
norm_sgc = avg_num3 
norm_bgc = avg_num3 
norm_issgd = avg_num3 
norm_ehd = avg_num3 
norm_od = avg_num3 