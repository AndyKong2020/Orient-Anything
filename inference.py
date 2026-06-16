import torch
from PIL import Image
from utils import *
import torch.nn.functional as F
import numpy as np

def _decode_3angle_prediction(dino_pred):
    dino_pred = dino_pred.detach().float().cpu()
    out_dim = dino_pred.shape[-1]
    rotation_bins = out_dim - (360 + 180 + 2)
    if rotation_bins not in (180, 360):
        raise ValueError(f"Unsupported prediction dimension: {out_dim}")

    gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
    gaus_pl_pred = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
    gaus_ro_pred = torch.argmax(dino_pred[:, 360+180:360+180+rotation_bins], dim=-1)
    confidence = F.softmax(dino_pred[:, -2:], dim=-1)[:, 0]
    rotation_offset = 90 if rotation_bins == 180 else 180
    return gaus_ax_pred, gaus_pl_pred, gaus_ro_pred, confidence, rotation_offset

def get_3angle(image, dino, val_preprocess, device):
    
    # image = Image.open(image_path).convert('RGB')
    image_inputs = val_preprocess(images = image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred, gaus_pl_pred, gaus_ro_pred, confidence, rotation_offset = _decode_3angle_prediction(dino_pred)
    angles = torch.zeros(4)
    angles[0]  = float(gaus_ax_pred[0])
    angles[1]  = float(gaus_pl_pred[0] - 90)
    angles[2]  = float(gaus_ro_pred[0] - rotation_offset)
    angles[3]  = float(confidence[0])
    return angles

def get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device):
    
    # image = Image.open(image_path).convert('RGB')
    image = get_crop_images(origin_img, num=3) + get_crop_images(rm_bkg_img, num=3)
    image_inputs = val_preprocess(images = image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred, gaus_pl_pred, gaus_ro_pred, confidence, rotation_offset = _decode_3angle_prediction(dino_pred)
    gaus_ax_pred = gaus_ax_pred.to(torch.float32)
    gaus_pl_pred = gaus_pl_pred.to(torch.float32)
    gaus_ro_pred = gaus_ro_pred.to(torch.float32)
    
    gaus_ax_pred   = remove_outliers_and_average_circular(gaus_ax_pred)
    gaus_pl_pred   = remove_outliers_and_average(gaus_pl_pred)
    gaus_ro_pred   = remove_outliers_and_average(gaus_ro_pred)
    
    confidence     = torch.mean(confidence)
    angles = torch.zeros(4)
    angles[0]  = gaus_ax_pred
    angles[1]  = gaus_pl_pred - 90
    angles[2]  = gaus_ro_pred - rotation_offset
    angles[3]  = confidence
    return angles
