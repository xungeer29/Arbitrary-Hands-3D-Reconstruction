import torch
import torch.nn as nn


class Cal_Loss(nn.CrossEntropyLoss):
    def __init__(self, wc=80, wp=160, wj3d=200, wpaj3d=360, wpj2d=400, wbone=200, wpose=80, wshape=10):
        super(Cal_Loss, self).__init__()
        # self.mano_layer = mano_layer
        self.wc, self.wp, self.wj3d, self.wpaj3d, self.wpj2d, self.wbone, self.wpose, self.wshape = wc, wp, wj3d, wpaj3d, wpj2d, wbone, wpose, wshape
        self.seg_loss = nn.CrossEntropyLoss(weight=None, reduce=None, reduction='mean', label_smoothing=0.0)
        # self.center_loss = nn.CrossEntropyLoss(weight=None, reduce=None, reduction='mean', label_smoothing=0.0)
        self.center_loss = nn.MSELoss(reduction='none')
        self.pose_loss = nn.MSELoss(reduction='none')
        self.shape_loss = nn.MSELoss(reduction='none')
        self.pj2d_loss = nn.MSELoss(reduction='none')
        self.j3d_loss = nn.MSELoss(reduction='none')
        self.bone_loss = nn.MSELoss(reduction='none')
        self.bone_idx_start = [0,1,2,3, 0,5,6,7, 0,9,10,11,  0, 13,14,15, 0, 17,18,19]
        self.bone_idx_end =   [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20]
    
    def forward(self, output, meta_data):
        seg_loss = self.seg_loss(output['segms'].sigmoid(), meta_data['segms'])
        
        l_center_flag = ((meta_data['center_map'][:,[0]]>0).float().sum(-1).sum(-1)>0).float()
        r_center_flag = ((meta_data['center_map'][:,[1]]>0).float().sum(-1).sum(-1)>0).float()
        # l_center_loss = (self.center_loss(output['l_center_map'].sigmoid(), meta_data['center_map'][:,[0]]).mean(-1).mean(-1) * l_center_flag).sum()/(l_center_flag.sum()+1e-12)
        # r_center_loss = (self.center_loss(output['r_center_map'].sigmoid(), meta_data['center_map'][:,[1]]).mean(-1).mean(-1) * r_center_flag).sum()/(r_center_flag.sum()+1e-12)
        l_center_loss = (self.center_loss(output['l_center_map'], meta_data['center_map'][:,[0]]).mean(-1).mean(-1) * l_center_flag).sum()/(l_center_flag.sum()+1e-12)
        r_center_loss = (self.center_loss(output['r_center_map'], meta_data['center_map'][:,[1]]).mean(-1).mean(-1) * r_center_flag).sum()/(r_center_flag.sum()+1e-12)
        center_loss = l_center_loss + r_center_loss

        l_pose_flag = (meta_data['param'][:, 3:3+96].abs().sum(-1)>0).float()
        r_pose_flag = (meta_data['param'][:, 3*2+96+10:3*2+96*2+10].abs().sum(-1)>0).float()
        l_pose_loss = (self.pose_loss(output['l_params_pred'][:, 3:3+96], meta_data['param'][:, 3:3+96]).mean(-1) * l_pose_flag).sum()/(l_pose_flag.sum()+1e-12)
        r_pose_loss = (self.pose_loss(output['r_params_pred'][:, 3:3+96], meta_data['param'][:, 3*2+96+10:3*2+96*2+10]).mean(-1) * r_pose_flag).sum()/(r_pose_flag.sum()+1e-12)
        pose_loss = l_pose_loss + r_pose_loss
        l_shape_loss = (self.shape_loss(output['l_params_pred'][:, 3+96:3+96+10], meta_data['param'][:, 3+96:3+96+10]).mean(-1) * l_pose_flag).sum()/(l_pose_flag.sum()+1e-12)
        r_shape_loss = (self.shape_loss(output['r_params_pred'][:, 3+96:3+96+10], meta_data['param'][:, 3*2+96*2+10:3*2+96*2+10*2]).mean(-1) * r_pose_flag).sum()/(r_pose_flag.sum()+1e-12)
        shape_loss = l_shape_loss + r_shape_loss
      
        # pj2d
        # weak persp
        l_j2d_pred = output['l_handJ'][:, :, :2] * output['l_params_pred'][:, [0]].unsqueeze(2) + output['l_params_pred'][:, 1:1+2].unsqueeze(1)
        r_j2d_pred = output['r_handJ'][:, :, :2] * output['r_params_pred'][:, [0]].unsqueeze(2) + output['r_params_pred'][:, 1:1+2].unsqueeze(1)
        l_j2d_pred = (l_j2d_pred + 1) / 2
        r_j2d_pred = (r_j2d_pred + 1) / 2
        l_pj2d_loss = (self.pj2d_loss(l_j2d_pred, meta_data['j2d'][:,:21]).mean(-1).mean(-1) * l_pose_flag).sum()/(l_pose_flag.sum()+1e-12)
        r_pj2d_loss = (self.pj2d_loss(r_j2d_pred, meta_data['j2d'][:, 21:]).mean(-1).mean(-1) * r_pose_flag).sum()/(r_pose_flag.sum()+1e-12)
        pj2d_loss = l_pj2d_loss + r_pj2d_loss

        # j3d
        l_j3d_loss = (self.j3d_loss(output['l_handJ'], meta_data['l_handJ']).mean(-1).mean(-1) * l_pose_flag).sum()/(l_pose_flag.sum()+1e-12)
        r_j3d_loss = (self.j3d_loss(output['r_handJ'], meta_data['r_handJ']).mean(-1).mean(-1) * r_pose_flag).sum()/(r_pose_flag.sum()+1e-12)
        j3d_loss = l_j3d_loss + r_j3d_loss

        # paj3d self.wpaj3d

        # bone self.wbone
        l_bone_len_pread = ((output['l_handJ'][:, self.bone_idx_start] - output['l_handJ'][:, self.bone_idx_end])**2).sum(-1)**0.5
        l_bone_len_gt = ((meta_data['l_handJ'][:, self.bone_idx_start] - meta_data['l_handJ'][:, self.bone_idx_end])**2).sum(-1)**0.5
        l_bone_loss = (self.bone_loss(l_bone_len_pread, l_bone_len_gt).mean(-1) * l_pose_flag).sum()/(l_pose_flag.sum()+1e-12)
        r_bone_len_pread = ((output['r_handJ'][:, self.bone_idx_start] - output['r_handJ'][:, self.bone_idx_end])**2).sum(-1)**0.5
        r_bone_len_gt = ((meta_data['r_handJ'][:, self.bone_idx_start] - meta_data['r_handJ'][:, self.bone_idx_end])**2).sum(-1)**0.5
        r_bone_loss = (self.bone_loss(r_bone_len_pread, r_bone_len_gt).mean(-1) * r_pose_flag).sum()/(r_pose_flag.sum()+1e-12)
        bone_loss = l_bone_loss + r_bone_loss

        loss = self.wp*seg_loss + self.wc*center_loss + self.wpose*pose_loss + self.wshape*shape_loss \
                + self.wpj2d*pj2d_loss + self.wj3d*j3d_loss + self.wbone*bone_loss

        return [loss, seg_loss, center_loss, pose_loss, shape_loss, pj2d_loss, j3d_loss, bone_loss]