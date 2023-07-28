import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# from acr.config import args
from acr.utils import vertices_kp3d_projection
from mano.manolayer import ManoLayer

align_idx = 9
mano_mesh_root_align = True
perspective_proj = False


class MANOWrapper(nn.Module):
    def __init__(self):
        super(MANOWrapper,self).__init__()
        self.mano_layer=nn.ModuleDict({
            'r':ManoLayer(
                    ncomps=45,
                    # center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    center_idx=align_idx if mano_mesh_root_align else None,
                    side='right',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            'l':ManoLayer(
                    ncomps=45,
                    # center_idx=args().align_idx if args().mano_mesh_root_align else None,
                    center_idx=align_idx if mano_mesh_root_align else None,
                    side='left',
                    mano_root='mano/',
                    use_pca=False,
                    flat_hand_mean=False,
                )
        })
        self.mano_layer['l'].th_shapedirs[:,0,:] *= -1

    def forward(self, outputs, meta_data):

        params_dict = outputs['params_dict']
        L, R = outputs['left_hand_num'], outputs['right_hand_num']
        outputs['output_hand_type'] = torch.cat((torch.zeros(L), torch.ones(R))).to(device=params_dict['poses'].device).to(torch.int32)

        l_vertices, l_joints, _ = self.mano_layer['l'](params_dict['poses'][:L], th_betas=params_dict['betas'][:L]) # if empty, return empty [1,48] [1,10]
        r_vertices, r_joints, _ = self.mano_layer['r'](params_dict['poses'][L:L+R], th_betas=params_dict['betas'][L:L+R])

        mano_outs = {'verts': torch.cat((l_vertices, r_vertices)), 'j3d':torch.cat((l_joints, r_joints))}
        outputs.update({**mano_outs})
        # outputs.update(vertices_kp3d_projection(outputs,params_dict=params_dict,meta_data=meta_data,presp=args().perspective_proj))        
        outputs.update(vertices_kp3d_projection(outputs,params_dict=params_dict,meta_data=meta_data,presp=perspective_proj))        

        return outputs
