import sys, os, cv2
# cv2.setNumThreads(1)
sys.path.append('.')
from tqdm import tqdm
from contextlib import suppress
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.utils import update_summary, ApexScaler, NativeScaler, distribute_bn
import warnings
warnings.filterwarnings('ignore')

import acr.lr_sc as StepLR_withWarmUp
from acr.utils import load_model, seed_everything
from acr.renderer.renderer_pt3d import get_renderer
from acr.model import ACR as ACR_v1
from acr.dataset import handDataset
from acr.trainer import train_one_epoch, validate
from acr.hparams import parse_args
from acr.losses import Cal_Loss

torch.backends.cudnn.benchmark = True

def main(args):
    model = ACR_v1()
    if os.path.exists(args.model_path):
        model = load_model(args.model_path, model, prefix = 'module.', drop_prefix='', fix_loaded=False)
        print(f'loaded {args.model_path}')
    
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print('Using native Torch AMP. Training in mixed precision.')

    args.distributed = False
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        args.distributed = True
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.local_rank = torch.distributed.get_rank()
        device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(device)
        device = torch.device('cuda:%d' % args.local_rank)
        print('Training in distributed mode: GPU %d, rank %d, world_size %d.' % (gpu_num, args.local_rank, args.world_size))
        model.to(device=device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[device], find_unused_parameters=True)
    else:
        device = torch.device('cuda:0')
        model.to(device=device)

    renderer = get_renderer(resolution=(512,512), perps=True, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0, amsgrad=False) # lr=5e-5

    # lr_scheduler = StepLR_withWarmUp(optimizer,
    #                                 #  last_epoch=-1 if cfg.TRAIN.current_epoch == 0 else cfg.TRAIN.current_epoch,
    #                                  init_lr=1e-3 * args.lr,
    #                                  warm_up_epoch=0,
    #                                  gamma=args.adjust_lr_factor,
    #                                  step_size=100,
    #                                  min_thres=0.05)
    lr_scheduler = StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    
    # dataset
    trainDataset = handDataset(interPath=args.inputs, train=True)
    # if args.exp == 'debug': trainDataset = trainDataset[:100]
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=train_sampler is None, 
                            sampler=train_sampler, batch_sampler=None, num_workers=args.nw, collate_fn=None,
                            pin_memory=True, drop_last=True)
    valDataset = handDataset(interPath=args.inputs, train=False)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False, 
                            sampler=None, batch_sampler=None, num_workers=args.nw, collate_fn=None,
                            pin_memory=True, drop_last=False)
    if args.exp == 'debug':
        trainLoader = valLoader

    loss_fn = Cal_Loss()

    best_metric = 1e6

    os.makedirs(f'logs/{args.exp}/', exist_ok=True)

    for epoch in range(0, args.epoch):
        if args.distributed and hasattr(trainLoader.sampler, 'set_epoch'):
            trainLoader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(epoch, model, trainLoader, optimizer, loss_fn, args,
                        lr_scheduler=lr_scheduler, renderer=renderer, saver=None, output_dir=None,
                        amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=None, mixup_fn=None,
                        device=device)
        if args.distributed:
            if args.local_rank == 0:
                print("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, reduce=True)
        # eval_metrics = validate(model, loader_eval, validate_loss_fn, args, renderer, amp_autocast=amp_autocast)
        eval_metrics = train_metrics

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step()

        if args.local_rank == 0:
            update_summary(
                epoch, train_metrics, eval_metrics, f'logs/{args.exp}/summary-epoch.csv',
                write_header=best_metric==1e6, log_wandb=False)
        
            if best_metric > eval_metrics['loss']:
                best_metric = eval_metrics['loss']
                torch.save(model.state_dict(), f'logs/{args.exp}/best.pth')
            torch.save(model.state_dict(), f'logs/{args.exp}/last.pth')    

# CUDA_VISIBLE_DEVICES=4 python -m acr.main --demo_mode video -t --inputs /data/gaofuxun/InterHand/upperbody/sample-video-4-3.mp4
if __name__ == '__main__':
    seed_everything(2023)
    hparams = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    if hparams.exp == 'debug':
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        torch.autograd.set_detect_anomaly(True)
        os.system(f'rm -rf logs/{hparams.exp}')
        hparams.batch_size = 8
        hparams.nw = 0
        hparams.log_interval = 1
    if not os.path.isdir(f'logs/{hparams.exp}'):
        os.makedirs(f'logs/{hparams.exp}', exist_ok=True)
    # code backup
    suffix = 0
    while os.path.exists(f'logs/{hparams.exp}/acr{suffix}.zip'):
        suffix += 1
    os.system(f'zip -r logs/{hparams.exp}/acr{suffix}.zip acr/ > /dev/null')
    main(hparams)
