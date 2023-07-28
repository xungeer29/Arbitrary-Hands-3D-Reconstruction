import json
import os.path as osp
from tqdm import tqdm
import pickle
from glob import glob
import os
import numpy as np

files = sorted(glob('/cache/gaofuxun/ACR/interhand2.6m/val/anno/*.pkl'))
# fpath = 'hand.csv'
# with open(fpath, 'w') as fw:
#     fw.write('cx,cy,w,h\n')

for f in tqdm(files):
    anno = pickle.load(open(f, 'rb'))
    img = anno['img']
    H, W, C = img.shape
    c2ds = anno['c2ds']
    if c2ds['left'] is not None and c2ds['right'] is not None:
        print(f)
        exit()
    # for hand_type in ['left', 'right']:
    #     if c2ds[hand_type] is None:
    #         continue
    #     cx, cy, w, h = c2ds[hand_type]
    #     print(c2ds[hand_type], H, W)
    #     with open(fpath, 'a') as fw:
    #         if cx > 1:
    #             fw.write(f'{cx/W},{cy/H},{w/W},{h/H}\n')
    #         else:
    #             fw.write(f'{cx},{cy},{w},{h}\n')
    
