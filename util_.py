
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from util_param_ import *


##############################################################################################################################################################
##############################################################################################################################################################

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as FF
def fig_plot(model, model_id = 0, img_id2 = img_id):
    model.eval().to(device)

    root     = './data/COCO/val2017/'
    img_path = f'{root}{str(img_id2).zfill(12)}.jpg'
    im       = Image.open(img_path)

    pred = model([FF.to_tensor(im).to(device)])[0]
    boxes, labels, scores = pred['boxes'], pred['labels'], pred['scores']

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 16)           
    COCO_CLASSES = [
        '__background__',   # 0
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench',
        'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
        'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
        'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
        'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
        'hair drier','toothbrush'                 # 80
    ]
    label_map = {i: name for i, name in enumerate(COCO_CLASSES)}

    for box, lab, sc in zip(boxes, labels, scores):
        if sc < 0.5: break
        x1,y1,x2,y2 = box.cpu().tolist()
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        draw.text((x1, y1), f'{label_map[int(lab)]} {sc:.2f}', fill='white', font=font)

    label_map = {i: name for i, name in enumerate(COCO_CLASSES)}

    for box, lab, sc in zip(boxes, labels, scores):
        if sc < 0.5: break
        x1,y1,x2,y2 = box.cpu().tolist()
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        draw.text((x1, y1), f'{label_map[int(lab)]} {sc:.2f}', fill='white', font=font)

    im.save('zvis_'+model_id+'.png', dpi=(300,300))
