import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
from timm.utils import AverageMeter, update_summary, reduce_tensor
from timm.models import model_parameters

from pytorch3d.transforms import rotation_6d_to_matrix
from acr.utils import rotation_matrix_to_angle_axis, rot6D_to_angular, visualize, save_obj


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, renderer=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, device=torch.device('cuda:0')):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    seg_loss_m, center_loss_m, pose_loss_m, shape_loss_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    pj2d_loss_m, j3d_loss_m, bone_loss_m = AverageMeter(), AverageMeter(), AverageMeter()

    model.train()

    start = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (ori_img, imgTensor, mask, hms, param, ks, j2d, c2d, batch_ids) in enumerate(loader):
        last_batch = batch_idx == last_idx
        meta_data = {}
        meta_data['image'] = imgTensor.to(device=device)
        meta_data['batch_ids'] = batch_ids
        meta_data['segms'] = mask.to(device=device)
        meta_data['center_map'] = hms.to(device=device)
        meta_data['param'] = param.to(device=device)
        meta_data['ks'] = ks.to(device=device)
        meta_data['j2d'] = j2d.to(device=device)
        meta_data['c2d'] = c2d.to(device=device)

        with amp_autocast():
            output = model(meta_data)

            # MANO
            bs = param.shape[0]
            pose_mat_l = rotation_6d_to_matrix(meta_data['param'][:, 3:3+96].reshape(-1, 6))
            l_pose_gt = rotation_matrix_to_angle_axis(pose_mat_l).reshape(bs, -1)
            l_handV_gt, l_handJ_gt, _ = loader.dataset.mano_layer['left'](l_pose_gt, th_betas=meta_data['param'][:, 3+96:3+96+10])
            pose_mat_r = rotation_6d_to_matrix(meta_data['param'][:, 3*2+96+10:3*2+96*2+10].reshape(-1, 6))
            r_pose_gt = rotation_matrix_to_angle_axis(pose_mat_r).reshape(bs, -1)
            r_handV_gt, r_handJ_gt, _ = loader.dataset.mano_layer['right'](r_pose_gt, th_betas=meta_data['param'][:, 3*2+96*2+10:3*2+96*2+10*2])
            meta_data['l_handV'], meta_data['l_handJ'] = l_handV_gt, l_handJ_gt
            meta_data['r_handV'], meta_data['r_handJ'] = r_handV_gt, r_handJ_gt

            # pred
            pose_mat_l = rotation_6d_to_matrix(output['l_params_pred'][:, 3:3+96].reshape(-1, 6))
            l_pose_pred = rotation_matrix_to_angle_axis(pose_mat_l).reshape(bs, -1)
            l_handV_pred, l_handJ_pred, _ = loader.dataset.mano_layer['left'](l_pose_pred, th_betas=output['l_params_pred'][:, 3+96:3+96+10])
            pose_mat_r = rotation_6d_to_matrix(output['r_params_pred'][:, 3:3+96].reshape(-1, 6))
            r_pose_pred = rotation_matrix_to_angle_axis(pose_mat_r).reshape(bs, -1)
            r_handV_pred, r_handJ_pred, _ = loader.dataset.mano_layer['right'](r_pose_pred, th_betas=output['r_params_pred'][:, 3+96:3+96+10])
            output['l_handV'], output['l_handJ'] = l_handV_pred, l_handJ_pred
            output['r_handV'], output['r_handJ'] = r_handV_pred, r_handJ_pred
            output['l_face'], output['r_face'] = loader.dataset.mano_layer['left'].th_faces, loader.dataset.mano_layer['right'].th_faces

            [loss, seg_loss, center_loss, pose_loss, shape_loss, pj2d_loss, j3d_loss, bone_loss] = loss_fn(output, meta_data)

        if not args.distributed:
            losses_m.update(loss.item(), imgTensor.size(0))
            seg_loss_m.update(seg_loss.item(), imgTensor.size(0))
            center_loss_m.update(center_loss.item(), imgTensor.size(0))
            pose_loss_m.update(pose_loss.item(), imgTensor.size(0))
            shape_loss_m.update(shape_loss.item(), imgTensor.size(0))
            pj2d_loss_m.update(pj2d_loss.item(), imgTensor.size(0))
            j3d_loss_m.update(j3d_loss.item(), imgTensor.size(0))
            bone_loss_m.update(bone_loss.item(), imgTensor.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            optimizer.step()
            
        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - start)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(seg_loss.data, args.world_size)
                seg_loss_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(center_loss.data, args.world_size)
                center_loss_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(pose_loss.data, args.world_size)
                pose_loss_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(shape_loss.data, args.world_size)
                shape_loss_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(j3d_loss.data, args.world_size)
                j3d_loss_m.update(reduced_loss.item(), imgTensor.size(0))
                reduced_loss = reduce_tensor(bone_loss.data, args.world_size)
                bone_loss_m.update(reduced_loss.item(), imgTensor.size(0))

            metrics = OrderedDict([('lr', lr), ('loss', losses_m.avg), ('seg_loss', seg_loss_m.avg), ('center_loss', center_loss_m.avg), ('pose_loss', pose_loss_m.avg),
                    ('shape_loss', shape_loss_m.avg), ('pj2d_loss', pj2d_loss_m.avg), ('j3d_loss', j3d_loss_m.avg), ('bone_loss', bone_loss_m.avg)])
            
            if args.local_rank == 0:
                print(
                    'Train: {} [{:>4d}/{}]  '
                    'LR: {lr:.3e}  '
                    'Loss: {loss.avg:#.5g}  '
                    'Seg: {seg_loss.avg:#.3g}  '
                    'Center: {center_loss.avg:#.3g}  '
                    'Pose: {pose_loss.avg:#.3g}  '
                    'Shape: {shape_loss.avg:#.3g}  '
                    'Pj2d: {pj2d_loss.avg:#.3g}  '
                    'J3d: {j3d_loss.avg:#.3g}  '
                    'Bone: {bone_loss.avg:#.3g}  '
                    'Time: {batch_time.val:#.3f}s  '
                    ''.format(
                        epoch, batch_idx, len(loader),
                        lr=lr,
                        loss=losses_m,
                        seg_loss=seg_loss_m,
                        center_loss=center_loss_m,
                        pose_loss=pose_loss_m,
                        shape_loss=shape_loss_m,
                        pj2d_loss=pj2d_loss_m,
                        j3d_loss=j3d_loss_m,
                        bone_loss=bone_loss_m,
                        batch_time=batch_time_m,
                        ))
            
                update_summary(
                    epoch, metrics, OrderedDict([]), f'logs/{args.exp}/summary-train.csv',
                    write_header=(epoch*len(loader)+batch_idx)==0, log_wandb=False)
                if last_batch or batch_idx % (args.log_interval * 10) == 0:
                    output['image'] = meta_data['image']
                    visualize(output, renderer, save_dir=f'logs/{args.exp}/', name=epoch*len(loader)+batch_idx)

        # if lr_scheduler is not None:
        #     lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)


    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return metrics


def validate(epoch, model, loader, loss_fn, args, renderer=None, amp_autocast=suppress, log_suffix='', device=torch.device('cuda:0')):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    seg_loss_m, center_loss_m, pose_loss_m, shape_loss_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    pj2d_loss_m, j3d_loss_m, bone_loss_m = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()

    start = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (ori_img, imgTensor, mask, hms, param, ks, j2d, c2d, batch_ids) in enumerate(loader):
        last_batch = batch_idx == last_idx
        meta_data = {}
        meta_data['image'] = imgTensor.to(device=device)
        meta_data['batch_ids'] = batch_ids
        meta_data['segms'] = mask.to(device=device)
        meta_data['center_map'] = hms.to(device=device)
        meta_data['param'] = param.to(device=device)
        meta_data['ks'] = ks.to(device=device)
        meta_data['j2d'] = j2d.to(device=device)
        meta_data['c2d'] = c2d.to(device=device)

        with amp_autocast():
            with torch.no_grad():
                output = model(meta_data)

            # MANO
            # pose_mat_l = rotation_6d_to_matrix(pose_l.view(-1, 6))
            # pose_l = rotation_matrix_to_angle_axis(pose_mat_l).reshape(-1)
            l_pose_gt = rot6D_to_angular(meta_data['param'][:, 3:3+96])
            l_handV_gt, l_handJ_gt, _ = loader.dataset.mano_layer['left'](l_pose_gt, th_betas=meta_data['param'][:, 3+96:3+96+10])
            r_pose_gt = rot6D_to_angular(meta_data['param'][:, 3*2+96+10:3*2+96*2+10])
            r_handV_gt, r_handJ_gt, _ = loader.dataset.mano_layer['right'](r_pose_gt, th_betas=meta_data['param'][:, 3*2+96*2+10:3*2+96*2+10*2])
            meta_data['l_handV'], meta_data['l_handJ'] = l_handV_gt, l_handJ_gt
            meta_data['r_handV'], meta_data['r_handJ'] = r_handV_gt, r_handJ_gt
            # pred
            l_pose_pred = rot6D_to_angular(output['l_params_pred'][:, 3:3+96])
            l_handV_pred, l_handJ_pred, _ = loader.dataset.mano_layer['left'](l_pose_pred, th_betas=output['l_params_pred'][:, 3+96:3+96+10])
            r_pose_pred = rot6D_to_angular(output['r_params_pred'][:, 3:3+96])
            r_handV_pred, r_handJ_pred, _ = loader.dataset.mano_layer['right'](r_pose_pred, th_betas=output['r_params_pred'][:, 3+96:3+96+10])
            output['l_handV'], output['l_handJ'] = l_handV_pred, l_handJ_pred
            output['r_handV'], output['r_handJ'] = r_handV_pred, r_handJ_pred

            [loss, seg_loss, center_loss, pose_loss, shape_loss, pj2d_loss, j3d_loss, bone_loss] = loss_fn(output, meta_data)

        if not args.distributed:
            losses_m.update(loss.item(), imgTensor.size(0))
            seg_loss_m.update(seg_loss.item(), imgTensor.size(0))
            center_loss_m.update(center_loss.item(), imgTensor.size(0))
            pose_loss_m.update(pose_loss.item(), imgTensor.size(0))
            shape_loss_m.update(shape_loss.item(), imgTensor.size(0))
            pj2d_loss_m.update(pj2d_loss.item(), imgTensor.size(0))
            j3d_loss_m.update(j3d_loss.item(), imgTensor.size(0))
            bone_loss_m.update(bone_loss.item(), imgTensor.size(0))

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - start)

    print(
        'EVAL: {}  '
        'Loss: {loss.avg:#.3g}  '
        'Seg: {seg_loss.avg:#.3g}  '
        'Center: {center_loss.avg:#.3g}  '
        'Pose: {pose_loss.avg:#.3g}  '
        'Shape: {shape_loss.avg:#.3g}  '
        'Pj2d: {pj2d_loss.avg:#.3g}  '
        'J3d: {j3d_loss.avg:#.3g}  '
        'Bone: {bone_loss.avg:#.3g}  '
        'Time: {batch_time.val:.3f}s  '
        ''.format(
            epoch,
            loss=losses_m,
            seg_loss=seg_loss_m,
            center_loss=center_loss_m,
            pose_loss=pose_loss_m,
            shape_loss=shape_loss_m,
            pj2d_loss=pj2d_loss_m,
            j3d_loss=j3d_loss_m,
            bone_loss=bone_loss_m,
            batch_time=batch_time_m,
            ))
    metrics = OrderedDict([('loss', losses_m.avg), ('seg_loss', seg_loss_m.avg), ('center_loss', center_loss_m.avg), ('pose_loss', pose_loss_m.avg),
            ('shape_loss', shape_loss_m.avg), ('pj2d_loss', pj2d_loss_m.avg), ('j3d_loss', j3d_loss_m.avg), ('bone_loss', bone_loss_m.avg)])
    update_summary(
        epoch, metrics, OrderedDict([]), f'logs/{args.exp}/summary-eval.csv',
        write_header=(epoch*len(loader))==0, log_wandb=False)
    return metrics


