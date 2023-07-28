import json
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np
import pickle
from glob import glob

import os
import sys

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from acr.utils import batch_rodrigues, rotation_matrix_to_angle_axis, save_obj, rot6D_to_angular
# from acr.config import args
from mano.manolayer import ManoLayer


align_idx = 9
mano_mesh_root_align = True

class handDataset(Dataset):
    """mix different hand datasets"""

    def __init__(self, 
                 interPath=None,
                 theta=[-90, 90], scale=[0.75, 1.25], uv=[-10, 10],
                 flip=True,
                 train=True,
                 bone_length=None,
                 noise=0.0):

        self.dataset = {}
        self.theta = theta
        self.scale = scale
        self.uv = uv
        self.noise = noise
        self.flip = flip

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        self.train = train
        self.bone_length = bone_length

        if self.train:
            split = 'train'
        else:
            split = 'val'
        self.dataset['interhand2.6m'] = InterHand_dataset(interPath, split)
        print('load interhand2.6m {} dataset, size: {}'.format(split, len(self.dataset['interhand2.6m'])))

        self.mano_layer=nn.ModuleDict({
            'right':ManoLayer(
                    ncomps=45,
                    # center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    center_idx=align_idx if mano_mesh_root_align else None,
                    side='right',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            'left':ManoLayer(
                    ncomps=45,
                    # center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    center_idx=align_idx if mano_mesh_root_align else None,
                    side='left',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                )
        }).cuda()
        self.mano_layer['left'].th_shapedirs[:,0,:] *= -1

    def __len__(self):
        size = 0
        for k in self.dataset.keys():
            size += len(self.dataset[k])
        return size
        # return 1000 # debug

    def augm_params(self):
        theta = random.random() * (self.theta[1] - self.theta[0]) + self.theta[0]
        scale = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        u = random.random() * (self.uv[1] - self.uv[0]) + self.uv[0]
        v = random.random() * (self.uv[1] - self.uv[0]) + self.uv[0]
        flip = random.random() > 0.5 if self.flip else False
        return theta, scale, u, v, flip

    def process_data(self, img, mask, heatmaps, sigmas, manos, J2ds, c2ds, weak_persp):
        # pad to 512
        height, width = img.shape[:2]
        if  height > width:
            height_ = 512
            width_ = int(512 / height * width)
            pl = int((512 - width_) / 2)
            pr = 512 - width_ - pl
            pu, pd = 0, 0
            scale = height_ / height
        else:
            width_ = 512
            height_ = int(512 / width * height)
            pl, pr = 0, 0
            pu = int((512 - height_) / 2)
            pd = 512 - height_ - pu
            scale = width_ / width
        img = cv2.resize(img, (width_, height_))
        img = np.pad(img, ((pu, pd), (pl, pr), (0, 0)), 'constant')
        
        mask = cv2.resize(mask, (width_, height_))
        mask = np.pad(mask, ((pu, pd), (pl, pr), (0, 0)), mode='constant', constant_values=((255, 255), (255, 255), (255, 255)))
        mask = cv2.resize(mask, (256, 256))
        mask = (np.round(mask.astype(np.float32)/15)).astype(np.int8)
        mask_ = np.zeros((33, 256, 256))
        assert len(np.unique(mask_[:,:,0]) < 10)
        for i in range(16):
            mask_[i] = (mask[:,:,1] == i).astype(np.float32)
            mask_[i+16] = (mask[:,:,2] == i).astype(np.float32)
        mask_[32] = ((mask[:,:,1] == 17) * (mask[:,:,2] == 17)).astype(np.float32)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # for i, m in enumerate(mask_):
        #     m = cv2.erode(m, kernel, iterations=2)
        #     mask_[i] = cv2.dilate(m, kernel, iterations=2)

        heatmap_l = heatmaps[0]
        heatmap_r = heatmaps[1]
        j2d_l = J2ds['left']
        j2d_r = J2ds['right']
        c2d_l = c2ds['left']
        c2d_r = c2ds['right']

        j2d = torch.zeros((21*2, 2))
        c2d = torch.zeros((1*2, 4))
        if j2d_l is not None:
            j2d_l += np.array([pl, pu])
            j2d_l /= 512
            c2d_l += np.array([pl, pu, pl, pu])
            c2d_l /= 512
            j2d[:21] = torch.from_numpy(j2d_l)
            c2d[:1] = torch.from_numpy(c2d_l)
        if j2d_r is not None:
            j2d_r += np.array([pl, pu])
            j2d_r /= 512
            c2d_r += np.array([pl, pu, pl, pu])
            c2d_r /= 512
            j2d[21:] = torch.from_numpy(j2d_r)
            c2d[1:] = torch.from_numpy(c2d_r)

        param = torch.zeros((218))
        k_l, k_r = 0, 0
        mano_l = manos['left']
        mano_r = manos['right']
        if mano_l is not None:
            pose_l, shape_l = mano_l['pose'], mano_l['shape']
            pose_mat_l = batch_rodrigues(torch.tensor(pose_l).view(-1, 3))
            pose6d_l = matrix_to_rotation_6d(pose_mat_l).view(-1)
            # scale and pad weak_persp
            weak_persp_l = weak_persp['left']
            weak_persp_l[0] *= scale
            tx, ty = pl / 256, pu / 256
            weak_persp_l[1] += tx
            weak_persp_l[2] += ty

            k_l = sigmas[0]

            # ['cam', 'global_orient', 'hand_pose', 'betas']
            param[:3] = torch.tensor(weak_persp_l)
            param[3:3+96] = pose6d_l
            param[3+96:3+96+10] = torch.tensor(shape_l)
            # param[:96] = pose6d_l
            # param[96:96+10] = torch.tensor(shape_l)
            # param[96+10:96+10+3] = torch.tensor(weak_persp_l)

        if mano_r is not None:
            pose_r, shape_r = mano_r['pose'], mano_r['shape']
            pose_mat_r = batch_rodrigues(torch.tensor(pose_r).view(-1, 3))
            pose6d_r = matrix_to_rotation_6d(pose_mat_r).view(-1)
            weak_persp_r = weak_persp['right']
            # scale and pad weak_persp
            weak_persp_r = weak_persp['right']
            weak_persp_r[0] *= scale
            tx, ty = pl / 256, pu / 256
            weak_persp_r[1] += tx
            weak_persp_r[2] += ty

            k_r = sigmas[-1]

            # ['cam', 'global_orient', 'hand_pose', 'betas']
            param[96+10+3:96+10+3*2] = torch.tensor(weak_persp_r)
            param[96+10+3*2:96*2+10+3*2] = pose6d_r
            param[96*2+10+3*2:] = torch.tensor(shape_r)
            # param[96+10+3:96*2+10+3] = pose6d_r
            # param[96*2+10+3:96*2+10*2+3] = torch.tensor(shape_r)
            # param[96*2+10*2+3:] = torch.tensor(weak_persp_r)
        
        # to torch tensor
        ori_img = torch.tensor(img, dtype=torch.float32) / 255
        ori_img = ori_img.permute(2, 0, 1)
        imgTensor = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        mask = torch.tensor(mask_, dtype=torch.float32)
        hms = torch.tensor(heatmaps, dtype=torch.float32)
        ks = torch.tensor([k_l, k_r], dtype=torch.float32)

        return ori_img, imgTensor, mask, hms, param, ks, j2d, c2d

    def __getitem__(self, idx):
        img, mask, heatmaps, sigmas, manos, J2ds, c2ds, weak_persp = self.dataset['interhand2.6m'][idx]
        ori_img, imgTensor, mask, hms, param, ks, j2d, c2d = self.process_data(img, mask, heatmaps, sigmas, manos, J2ds, c2ds, weak_persp)

        return ori_img, imgTensor, mask, hms, param, ks, j2d, c2d, idx

    def __visualize__(self, idx):
        ori_img, imgTensor, mask, hms, param, ks, j2d, c2d = self.__getitem__(idx)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        imgTensor = (imgTensor*torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)).permute(1, 2, 0) * 255
        img = np.clip(imgTensor.detach().numpy(), 0, 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(f'img_{idx}.jpg', img)
        # mask
        mask_ = np.zeros((512, 512, 3))
        for i, m in enumerate(mask.detach().numpy()):
            m = cv2.resize(m, (512, 512))
            if i < 16:
                mask_[:,:,0] += i*15*m
            elif i < 32:
                mask_[:,:, 2] += (i-16)*15*m
            else:
                mask_[:,:,1] += 255*m
        assert np.max(mask_) <= 255, f'max: {np.max(mask_)}'
        assert len(np.unique(mask_[:,:,0])) <= 16, f'{np.unique(mask_[:,:,0])}'
        assert len(np.unique(mask_[:,:,2])) <= 16, f'{np.unique(mask_[:,:,2])}'
        mask = img*0.8 + mask_*0.2
        cv2.imwrite(f'mask_{idx}.jpg', mask)
        # hms
        hms_ = np.zeros((64, 64, 3))
        hms = hms.detach().numpy()
        hms_[:,:,0] = (hms[0]*255).astype(np.uint8)
        hms_[:,:,2] = (hms[1]*255).astype(np.uint8)
        hms_ = cv2.resize(hms_, (512, 512))
        hms = img*0.7 + hms_*0.3
        cv2.imwrite(f'hms_{idx}.jpg', hms)
        # j2d
        img_j2d = img.copy()
        for i, j in enumerate(j2d.detach().numpy()):
            cv2.circle(img_j2d, (int(j[0]*512), int(j[1]*512)), 1, (255,0,0), 1)
            cv2.putText(img_j2d, str(i), (int(j[0]*512), int(j[1]*512)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        # c2d
        for i, j in enumerate(c2d.detach().numpy()):
            cv2.circle(img_j2d, (int(j[0]*512), int(j[1]*512)), 1, (0,255,0), 1)
            cv2.putText(img_j2d, 'C', (int(j[0]*512), int(j[1]*512)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(f'j2d_{idx}.jpg', img_j2d)

        # mano
        pose_l, beta_l, weak_l = param[:96], param[96:96+10], param[96+10:96+10+3]
        pose_r, beta_r, weak_r = param[96+10+3:96*2+10+3], param[96*2+10+3:96*2+10*2+3], param[96*2+10*2+3:]
        img_weak = img.copy()
        if beta_l.sum() != 0:
            pose_mat_l = rotation_6d_to_matrix(pose_l.view(-1, 6))
            pose_l = rotation_matrix_to_angle_axis(pose_mat_l).reshape(-1)
            handV_l, handJ_l, _ = self.mano_layer['left'](pose_l.unsqueeze(0).cuda(), th_betas=beta_l.unsqueeze(0).cuda())
            face = self.mano_layer['left'].th_faces
            save_obj(handV_l[0].cpu().detach().numpy(), face.cpu().detach().numpy(), color=[0.5,0.5,0.5], obj_mesh_name='left.obj')
            # weak persp
            handJ2d = handJ_l[0, :, :2].cpu() * weak_l[0] + weak_l[1:]
            handJ2d = (handJ2d.detach().numpy() + 1) * 256
            for i, j in enumerate(handJ2d):
                cv2.circle(img_weak, (int(j[0]), int(j[1])), 1, (255,0,0), 1)
                cv2.putText(img_weak, str(i), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)

        if beta_r.sum() != 0:
            pose_mat_r = rotation_6d_to_matrix(pose_r.view(-1, 6))
            pose_r = rotation_matrix_to_angle_axis(pose_mat_r).reshape(-1)
            handV_r, handJ_r, _ = self.mano_layer['right'](pose_r.unsqueeze(0).cuda(), th_betas=beta_r.unsqueeze(0).cuda())
            face = self.mano_layer['right'].th_faces
            save_obj(handV_r[0].cpu().detach().numpy(), face.cpu().detach().numpy(), color=[0.5,0.5,0.5], obj_mesh_name='right.obj')
            # weak persp
            handJ2d = handJ_r[0, :, :2].cpu() * weak_r[0] + weak_r[1:]
            handJ2d = (handJ2d.detach().numpy() + 1) * 256
            for i, j in enumerate(handJ2d):
                cv2.circle(img_weak, (int(j[0]), int(j[1])), 1, (255,0,0), 1)
                cv2.putText(img_weak, str(i), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(f'weak_{idx}.jpg', img_weak)
       

class InterHand_dataset():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.img_root = '/data/gaofuxun/Data/InterHand2.6M/'

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            anno = pickle.load(file)
        img_path = anno['img_path']
        img = cv2.imread(osp.join(self.img_root, 'InterHand2.6M_5fps_batch1/images/', self.split, img_path))
        mask = cv2.imread(osp.join(self.data_path, self.split, 'mask', '{}.png'.format(idx)))
        heatmaps = anno['heatmaps']
        sigmas = anno['sigma']
        manos = anno['mano_dict']
        J2ds = anno['J2ds']
        c2ds = anno['c2ds']
        weak_persp = anno['weak_persp']

        return img, mask, heatmaps, sigmas, manos, J2ds, c2ds, weak_persp


if __name__ == '__main__':
    dataset = handDataset(interPath='/data/gaofuxun/Data/ACR/interhand2.6m', train=False)
    dataset.__visualize__(0)

