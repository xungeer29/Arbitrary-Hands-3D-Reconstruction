import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import pickle
from glob import glob
import os
import cv2

from mano.manolayer import ManoLayer
from acr.config import args
from acr.visualization import Visualizer, mano_two_hands_renderer
from acr.mano_wrapper import MANOWrapper
from acr.renderer.renderer_pt3d import get_renderer
from acr.utils import save_obj, generate_heatmap, rotation_matrix_to_angle_axis, rotation_matrix_to_quaternion, \
                    eulerAngles2rotationMat, axis_angle_to_quaternion, quaternion_to_angle_axis, batch_rodrigues


class InterHandLoader():
    def __init__(self, data_path, split='train'):
        assert split in ['train', 'test', 'val']

        self.root_path = data_path
        self.img_root_path = os.path.join(self.root_path, 'InterHand2.6M_5fps_batch1/images')
        self.annot_root_path = os.path.join(self.root_path, 'annotations')

        self.render_size = 512
        self.visualizer = Visualizer(resolution=(512, 512), renderer_type='pytorch3d')

        self.split = split

        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_data.json')) as f:
            self.data_info = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_camera.json')) as f:
            self.cam_params = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_MANO_NeuralAnnot.json')) as f:
            self.mano_params = json.load(f)

        self.data_size = len(self.data_info['images'])

        self.mano_layer=nn.ModuleDict({
            'right':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    side='right',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            'left':ManoLayer(
                    ncomps=45,
                    center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    side='left',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                )
        }).cuda()
        self.mano_layer['left'].th_shapedirs[:,0,:] *= -1

    def __len__(self):
        return self.data_size

    def show_data(self, idx):
        for k in self.data_info['images'][idx].keys():
            print(k, self.data_info['images'][idx][k])
        for k in self.data_info['annotations'][idx].keys():
            print(k, self.data_info['annotations'][idx][k])

    def load_camera(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        cam_idx = img_info['camera']

        capture_idx = str(capture_idx)
        cam_idx = str(cam_idx)
        cam_param = self.cam_params[str(capture_idx)]
        cam_t = np.array(cam_param['campos'][cam_idx], dtype=np.float32).reshape(3)
        cam_R = np.array(cam_param['camrot'][cam_idx], dtype=np.float32).reshape(3, 3)
        cam_t = -np.dot(cam_R, cam_t.reshape(3, 1)).reshape(3) / 1000  # -Rt -> t

        # add camera intrinsics
        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
        cameraIn = np.array([[focal[0], 0, princpt[0]],
                             [0, focal[1], princpt[1]],
                             [0, 0, 1]])
        return cam_R, cam_t, cameraIn

    def load_mano(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        frame_idx = img_info['frame_idx']
        capture_idx = str(capture_idx)

        cam_R, cam_t, cameraIn = self.load_camera(idx)

        frame_idx = str(frame_idx)
        mano_dict = {}
        coord_dict = {}
        for hand_type in ['left', 'right']:
            try:
                mano_param = self.mano_params[capture_idx][frame_idx][hand_type] # ['pose', 'shape', 'trans']
                mano_pose = torch.FloatTensor(mano_param['pose']).view(-1) # [48]
                mano_beta = torch.FloatTensor(mano_param['shape']).view(-1) # [10]
                trans = torch.FloatTensor(mano_param['trans'])

                handV, handJ, _ = self.mano_layer[hand_type](mano_pose.unsqueeze(0).cuda(), th_betas=mano_beta.unsqueeze(0).cuda(), th_trans=trans.unsqueeze(0).cuda())

                # 相机旋转施加到MANO系数的腕部(root)
                root_mat = batch_rodrigues(mano_pose[:3].unsqueeze(0))[0].numpy()
                root_mat = cam_R @ root_mat
                root_angle = rotation_matrix_to_angle_axis(torch.from_numpy(root_mat).unsqueeze(0))
                mano_pose[:3] = root_angle[0]
                handV_notrans, handJ_notrans, _ = self.mano_layer[hand_type](mano_pose.unsqueeze(0).cuda(), th_betas=mano_beta.unsqueeze(0).cuda())
                coord_dict[hand_type] = {'verts': handV[0].cpu().detach().numpy(), 'joints': handJ[0].cpu().detach().numpy(),
                                            'verts_notrans': handV_notrans[0].cpu().detach().numpy(), 'joints_notrans': handJ_notrans[0].cpu().detach().numpy()
                                            }
                mano_dict[hand_type] = {'pose': mano_pose.numpy(), 'shape': mano_beta.numpy(), 'trans': trans.numpy()}

            except:
                mano_dict[hand_type] = None
                coord_dict[hand_type] = None
        
        # handJ2d = self.Projection_block(handJ, cam_t, focal, princpt)
        # print(handJ2d);exit()

        return mano_dict, coord_dict

    def load_img(self, idx):
        img_info = self.data_info['images'][idx]
        img = cv.imread(osp.join(self.img_root_path, self.split, img_info['file_name']))
        return img, img_info['file_name']

def select_data(DATA_PATH, save_path, split):
    loader = InterHandLoader(DATA_PATH, split=split)
    # loader.show_data(55208);exit()
    render_data(loader, opt.save_path, split)

def render_data(loader, save_path, split):
    os.makedirs(osp.join(save_path, split, 'anno'), exist_ok=True)
    os.makedirs(osp.join(save_path, split, 'mask'), exist_ok=True)
    # os.makedirs(osp.join(save_path, split, 'heatmap'), exist_ok=True)
    # os.makedirs(osp.join(save_path, split, 'landmark'), exist_ok=True)

    renderer = mano_two_hands_renderer(img_size=512, device='cuda')
    # renderer_acr = get_renderer(resolution=(512, 512), perps=True)
    # camtrans_acr=renderer_acr.cameras.get_full_projection_transform()

    # test: 849160  val: 380125  train: 1361062
    processed = os.listdir(f'/lts0/InterHand/ACR/interhand2.6m/{split}/anno/')
    for idx in tqdm(range(120*10000, len(loader))):
    # for idx in tqdm(range(100*10000, 120*10000)):
        # loader.show_data(idx)
        if f'{idx}.pkl' in processed: continue
        anno = {}
        R, T, camera = loader.load_camera(idx)
        mano_dict, coord_dict = loader.load_mano(idx)
        # if coord_dict['left'] is None or coord_dict['right'] is None:
        #     continue
        # else:
        #     print('two hands!!!!!!!!!!!!!!!!!!!!!')
        #     loader.show_data(idx)
        # loader.show_data(idx)
        img, img_path = loader.load_img(idx)
        anno['mano_dict'] = mano_dict
        anno['img_path'] = img_path

        h, w, c = img.shape
        verts, faces, J2ds, c2ds = {'left': None, 'right': None}, {'left': None, 'right': None}, {'left': None, 'right': None}, {'left': None, 'right': None}
        weak_persp = {'left': None, 'right': None}
        for hand_type in ['left', 'right']:
            if coord_dict[hand_type] is None:
                continue
            
            handV = coord_dict[hand_type]['verts']
            handJ = coord_dict[hand_type]['joints']
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]
            # center
            mcp = handJ2d[[2,5,9,13,17]]
            mcp[:,0] = np.clip(mcp[:,0], 1, w-1)
            mcp[:,1] = np.clip(mcp[:,1], 1, h-1)
            c2d = mcp.mean(0)
            handV2d[:,0] = np.clip(handV2d[:,0], 1, w-1)
            handV2d[:,1] = np.clip(handV2d[:,1], 1, h-1)
            minx, miny, maxx, maxy = max(0, min(handV2d[:,0])), max(0,min(handV2d[:,1])), min(w,max(handV2d[:,0])), min(h, max(handV2d[:,1]))
            c2ds[hand_type] = np.array([c2d[0], c2d[1], maxx-minx, maxy-miny])

            verts[hand_type] = torch.from_numpy(handV).float().cuda().unsqueeze(0)
            faces[hand_type] = loader.mano_layer[hand_type].th_faces

            # Weak perspective projection
            handV_notrans = coord_dict[hand_type]['verts_notrans']
            handJ_notrans = coord_dict[hand_type]['joints_notrans']

            handJ2d_notrans = handJ_notrans[:,:2]
            handJ2d = handJ2d / np.array([[256, 256]]) - 1
            len1 = np.sqrt(np.sum((handJ2d[0]- handJ2d[1])**2))
            len2 = np.sqrt(np.sum((handJ2d_notrans[0]- handJ2d_notrans[1])**2))
            scale = len1 / len2
            [tx, ty] = handJ2d[0] - (handJ2d_notrans*scale)[0]
            weak_persp[hand_type] = [scale, tx, ty]
            handJ2d_notrans = handJ2d_notrans*scale + np.array([tx, ty])
            J2ds[hand_type] = (handJ2d_notrans+1) * np.array([[256, 256]])

            # img_ = np.zeros((512, 512, 3))
            # img_[:h,:w] = img
            # for i, j in enumerate((handJ2d_notrans+1) * np.array([[256, 256]])):
            #     cv.circle(img_, (int(j[0]), int(j[1])), 1, (255,0,0), 1)
            #     cv2.putText(img_, str(i), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
            # cv2.imwrite('tmp.png', img_[:h,:w])

        if verts['left'] is None and verts['right'] is None:
            continue

        # img_mask = renderer.render_mask(cameras=torch.from_numpy(camera).float().cuda().unsqueeze(0),
        #                                 v3d_left=verts['left'], v3d_right=verts['right'], 
        #                                 faces_left=faces['left'], faces_right=faces['right'])
        # img_mask = img_mask.detach().cpu().numpy()[0]
        # cv.imwrite(osp.join(save_path, split, 'mask', '{}.jpg'.format(idx)), img_mask[:h, :w])

        # 生成双手数据
        # if verts['left'] is None:
        #     verts['left'] = verts['right'] + torch.tensor([0.1, 0.2, 0.3]).to(verts['right'].device)
        # if verts['right'] is None:
        #     verts['right'] = verts['left'] + torch.tensor([0.1, 0.2, 0.3]).to(verts['left'].device)

        if verts['right'] is not None:
            v778 = (verts['right'][:, [119]] + verts['right'][:, [279]]) / 2
            verts['right'] = torch.cat((verts['right'], v778), 1)
        if verts['left'] is not None:
            v778 = (verts['left'][:, [119]] + verts['left'][:, [279]]) / 2
            verts['left'] = torch.cat((verts['left'], v778), 1)
        img_part_mask = renderer.render_part_mask(cameras=torch.from_numpy(camera).float().cuda().unsqueeze(0),
                                        v3d_left=verts['left'], v3d_right=verts['right']) 
        img_part_mask = img_part_mask.detach().cpu().numpy()[0]
        img_part_mask =  img_part_mask / img_part_mask.max() * 255
        img_part_mask_gray = cv2.cvtColor(img_part_mask, cv2.COLOR_BGR2GRAY)
        area = (img_part_mask_gray < 250).astype(np.uint8).sum()
        if area < 3000:
            continue
        cv.imwrite(osp.join(save_path, split, 'mask', '{}.png'.format(idx)), img_part_mask[:h, :w])
                
        anno['J2ds'] = J2ds
        anno['c2ds'] = c2ds
        # anno['mask'] = img_mask[:h, :w]
        # anno['part_mask'] = img_part_mask[:h, :w]
        
        # for hand_type in ['left', 'right']:
        #     j2d = J2ds[hand_type]
        #     if j2d is None:
        #         continue
        #     for i, j in enumerate(j2d):
        #         cv.circle(img, (int(j[0]), int(j[1])), 1, (255,0,0), 1)
        #         cv2.putText(img, str(i), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        #     c2d = c2ds[hand_type]
        #     cv.circle(img, (int(c2d[0]), int(c2d[1])), 1, (0,255,0), 1)
        # cv.imwrite(osp.join(save_path, split, 'landmark', '{}.jpg'.format(idx)), img)
        # save_obj(verts[0].cpu().detach().numpy(), faces[0].cpu().detach().numpy(), color=[0.5,0.5,0.5], obj_mesh_name='mesh.obj')
        try:
            heatmaps, (height, width, pl, pr, pu, pd), sigma = generate_heatmap(c2ds['left'], c2ds['right'], h, w, adaptive_sigma=True, padding=True)
        except:
            print(idx)
            loader.show_data(idx)
            print(c2ds)
            continue

        anno['heatmaps'] = heatmaps
        anno['sigma'] = sigma
        anno['heatmaps_pad'] = (height, width, pl, pr, pu, pd)
        anno['weak_persp'] = weak_persp
        
        # img_ = cv2.resize(img, (width, height))
        # img_ = np.pad(img_, ((pu, pd), (pl, pr), (0,0)), 'constant')
        # for heatmap in (heatmaps*255).astype(np.uint8):
        #     colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     img_ = colored_heatmap*0.2 + img_*0.8
        # img_ = cv2.resize(img_, (128, 128))
        # cv2.imwrite(osp.join(save_path, split, 'heatmap', '{}.jpg'.format(idx)), img_)

        with open(osp.join(save_path, split, 'anno', '{}.pkl'.format(idx)), 'wb') as handle:
            pickle.dump(anno, handle)


if __name__ == '__main__':
    import os 
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/lts0/InterHand/InterHand2.6M')
    parser.add_argument("--save_path", type=str, default='/lts0/InterHand/ACR/interhand2.6m')
    # parser.add_argument("--save_path", type=str, default='/cache/gaofuxun/ACR/interhand2.6m')
    opt = parser.parse_args()

    select_data(opt.data_path, opt.save_path, split='train')

    # for split in ['train', 'test', 'val']:
    #     select_data(opt.data_path, opt.save_path, split=split)

