import os, sys
import argparse
import yaml
import time

# currentfile=os.path.abspath(__file__)
# code_dir=currentfile.replace('config.py', '')

# project_dir=currentfile.replace(os.sep + 'acr' + os.sep + 'config.py', '')
# source_dir=currentfile.replace(os.sep + 'config.py', '')
# root_dir=project_dir.replace(project_dir.split(os.sep)[-1], '')

# time_stamp=time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(int(round(time.time()*1000))/1000))
# yaml_timestamp=os.path.abspath(os.path.join(project_dir, 'active_configs' + os.sep + "active_context_{}.yaml".format(time_stamp).replace(":", "_")))

# model_dir=os.path.join(project_dir, 'model_data')
# trained_model_dir=os.path.join(project_dir, 'trained_models')

def parse_args():
    def str2bool(str):
        return True if (str.lower() == 'True' or str.lower() == 'true') else False
    parser=argparse.ArgumentParser(description='ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction [CVPR 2023]')
    parser.add_argument('--inputs', default='/data/gaofuxun/Data/ACR/interhand2.6m', type=str, help='path to inputs') 
    parser.add_argument('--output_dir', type=str, help='path to save outputs') 
    parser.add_argument('--interactive_vis', action='store_true', help='whether to show the results in an interactive mode')
    parser.add_argument('--temporal_optimization', '-t', action='store_true', help='whether to optimize the temporal smoothness')
    # basic training settings
    parser.add_argument('--lr', help='lr', default=3e-4, type=float)
    parser.add_argument('--adjust_lr_factor', type=float, default=0.1, help='factor for adjusting the lr')
    parser.add_argument('--weight_decay', help='weight_decay', default=1e-6, type=float)
    parser.add_argument('--epoch', type=int, default=120, help='training epochs')
    parser.add_argument('--fine_tune', type=bool, default=True, help='whether to run online')
    parser.add_argument('-bs', '--batch_size', default=64*2, help='batch_size', type=int)
    parser.add_argument('--input_size', default=512, type=int, help='size of input image')
    parser.add_argument('--nw', default=16, help='number of workers', type=int)
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='choice of optimizer')
    parser.add_argument('--pretrain', type=str, default='simplebaseline', help='imagenet or spin or simplebaseline')
    parser.add_argument('--fix_backbone_training_scratch', type=bool, default=False, help='whether to fix the backbone features if we train the model from scratch.')
    parser.add_argument('--stop_seg_epoch', default=2, help='stop', type=int)
    parser.add_argument('--first_decay', default=10, help='stop', type=int)
    parser.add_argument('--second_decay', default=20, help='stop', type=int)
    parser.add_argument('--amp', default=True, help='whether to use mixed-precision (AMP)', type=bool)

    # loss settings
    parser.add_argument('--loss_thresh', default=1000, type=float, help='max loss value for a single loss')
    parser.add_argument('--max_supervise_num', default=-1, type=int, help='max hand number supervised in each batch for stable GPU memory usage')
    parser.add_argument('--supervise_cam_params', type=bool, default=False)
    parser.add_argument('--match_preds_to_gts_for_supervision', type=bool, default=True, help='whether to match preds to gts for supervision')
    parser.add_argument('--matching_mode', type=str, default='all', help='all | random_one | ')
    parser.add_argument('--supervise_global_rot', type=bool, default=False, help='whether supervise the global rotation of the estimated mano model')
    parser.add_argument('--HMloss_type', type=str, default='MSE', help='supervision for 2D pose heatmap: MSE or focal loss')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    # basic log settings
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--model_path', type=str, default ='./checkpoints/wild.pkl', help='trained model path')
    parser.add_argument('--exp', type=str, default='amp-ddp', help='Path to save log file')

    # augmentation settings
    parser.add_argument('--shuffle_crop_mode', type=str2bool, default=False, help='whether to shuffle the data loading mode between crop / uncrop for indoor 3D pose dataset only')
    parser.add_argument('--shuffle_crop_ratio_3d', default=0.9, type=float, help='the probability of changing the data loading mode from uncrop multi_hand to crop single hand')
    parser.add_argument('--shuffle_crop_ratio_2d', default=0.1, type=float, help='the probability of changing the data loading mode from uncrop multi_hand to crop single hand')
    parser.add_argument('--Synthetic_occlusion_ratio', default=0, type=float, help='whether to use use Synthetic occlusion')
    parser.add_argument('--color_jittering_ratio', default=0.2, type=float, help='whether to use use color jittering')
    parser.add_argument('--rotate_prob', default=0.2, type=float, help='whether to use rotation augmentation')
    parser.add_argument('--flip_ratio', default=0.5, type=float, help='whether to use rotation augmentation')

    parsed_args=parser.parse_args()
    
    return parsed_args
