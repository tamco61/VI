import matplotlib.pyplot as plt
from matplotlib import animation

import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

# rc('animation', html='jshtml')

from core.utils import to_tensors


# global variables
w, h = 432, 240
ref_length = 10  # ref_step
num_ref = -1
neighbor_stride = 5


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video
def read_frame_from_videos(video_path):
    vname = video_path
    frames = []
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst]
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w, h)))
    return frames

# set up models
device = torch.device("cpu")
net = importlib.import_module('model.e2fgvi')
model = net.InpaintGenerator().to(device)
ckpt_path = 'release_model/E2FGVI-CVPR22.pth'
data = torch.load(ckpt_path, map_location=device)
model.load_state_dict(data)
print(f'Loading model from: {ckpt_path}')
model.eval()

# prepare dataset
video_path = 'examples/batman'
mask_path = 'examples/batman_mask'
print(f'Loading videos and masks from: {video_path}')
frames = read_frame_from_videos(video_path)
video_length = len(frames)
imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
frames = [np.array(f).astype(np.uint8) for f in frames]

masks = read_mask(mask_path)
binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                for m in masks]
masks = to_tensors()(masks).unsqueeze(0)
imgs, masks = imgs.to(device), masks.to(device)
comp_frames = [None] * video_length

# completing holes by e2fgvi
print(f'Start test...')
for f in tqdm(range(0, video_length, neighbor_stride)):
    neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
    ref_ids = get_ref_index(f, neighbor_ids, video_length)
    selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
    selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
    with torch.no_grad():
        masked_imgs = selected_imgs*(1-selected_masks)
        pred_img, _ = model(masked_imgs, len(neighbor_ids))

        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            img = np.array(pred_img[i]).astype(
                np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx].astype(
                    np.float32)*0.5 + img.astype(np.float32)*0.5

# saving videos
print('Saving videos...')
save_dir_name = 'results'
ext_name = '_results.mp4'
save_base_name = video_path.split('/')[-1]
save_name = save_base_name + ext_name
if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)
save_path = os.path.join(save_dir_name, save_name)
writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        24, (w, h))
for f in range(video_length):
    comp = comp_frames[f].astype(np.uint8)
    writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
writer.release()
print(f'Finish test! The result video is saved in: {save_path}.')