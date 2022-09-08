from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter

sys.path.append('../SpatioTemporal Correlation Learning')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_provider import Provider, Validation
from utils.show import show_affs, val_show, show_affs_emb
from model.unet_residual import ResidualUNet
from model.unet_attention import AttentionUNet
from utils.utils import setup_seed, update_ema_variables
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss, BCE_loss_func
from loss.loss_embedding_mse import embedding_loss
from loss.loss_discriminative import discriminative_loss
from utils.evaluate import BestDice, AbsDiffFGLabels
from utils.seg_mutex import seg_mutex
from utils.affinity_stmap import multi_offset
from utils.postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel

from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Validation(cfg, mode='validation')
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer, ema=False):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    if  cfg.TRAIN.model_name == "AttentionUnet":

        model = ResidualUNet(in_channels=cfg.MODEL.input_nc,
                            out_channels=cfg.MODEL.output_nc,
                            nfeatures=cfg.MODEL.filters,
                            emd=cfg.MODEL.emd,
                            if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
    else:
        # "ResUnet"
        model = ResidualUNet(in_channels=cfg.MODEL.input_nc,
                                    out_channels=cfg.MODEL.output_nc,
                                    nfeatures=cfg.MODEL.filters,
                                    emd=cfg.MODEL.emd,
                                    if_sigmoid=cfg.MODEL.if_sigmoid,
                                    show_feature = cfg.MODEL.show_feature).to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)

    if ema:
        print('EMA...', end='', flush=True)
        for param in model.parameters():
            param.detach_()

    if cfg.MODEL.finetuning:
        print('finetuning...')
        model_path = os.path.join(cfg.TRAIN.save_path, cfg.MODEL.model_name, 'model-%06d.ckpt' % cfg.MODEL.model_id)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_weights'])

    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, ema_model, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_embedding = 0.0
    sum_loss_mask = 0.0
    device = torch.device('cuda:0')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
    nb_half = cfg.DATA.neighbor // 2

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    criterion_mask = BCE_loss_func
    criterion_ct = MSELoss()
    valid_mse = MSELoss()
    valid_bce = BCELoss()

    lr_strategies =['steplr', 'multi_steplr', 'explr', 'lambdalr']
    if cfg.TRAIN.lr_mode == 'steplr':
        print('Step LR')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.step_size, gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'multi_steplr':
        print('Multi step LR')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 150000], gamma=cfg.TRAIN.gamma)
    elif cfg.TRAIN.lr_mode == 'explr':
        print('Exponential LR')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif cfg.TRAIN.lr_mode == 'lambdalr':
        print('Lambda LR')
        lambda_func = lambda epoch: (1.0 - epoch / cfg.TRAIN.total_iters) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    else:
        print('Other LR scheduler')

    if cfg.TRAIN.deep_weight == 1:
        deep_weight_factor = [1.0, 1.0, 1.0, 1.0, 1.0]
    elif cfg.TRAIN.deep_weight == 2:
        deep_weight_factor = [0.01, 0.03, 0.1, 0.3, 1.0]
    else:
        deep_weight_factor = [cfg.TRAIN.deep_weight, 1.0, 1.0, 1.0, 1.0]

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        # ema_model.train()
        iters += 1
        t1 = time.time()
        batch_data = train_provider.next()
        inputs = batch_data['image'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()
        neighbor_mask = batch_data['neighbor'].cuda()

        down1 = batch_data['down1'].cuda()
        down2 = batch_data['down2'].cuda()
        down3 = batch_data['down3'].cuda()
        down4 = batch_data['down4'].cuda()
      
        if cfg.TRAIN.lr_mode in lr_strategies:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = cfg.TRAIN.base_lr
        
        optimizer.zero_grad()

        emd4, emd3, emd2, emd1, embedding, pred_mask = model(inputs)

        ##############################
        # LOSS
        # loss = criterion(pred, target, weightmap)
        loss_emd1, _, _ = embedding_loss(emd1, down1[:,0:nb_half*4], down1[:,nb_half*4:nb_half*8], down1[:,nb_half*8:nb_half*12], criterion, offsets[:nb_half*4], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        loss_emd2, _, _ = embedding_loss(emd2, down2[:,0:nb_half*3], down2[:,nb_half*3:nb_half*6], down2[:,nb_half*6:nb_half*9], criterion, offsets[:nb_half*3], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        loss_emd3, _, _ = embedding_loss(emd3, down3[:,0:nb_half*2], down3[:,nb_half*2:nb_half*4], down3[:,nb_half*4:nb_half*6], criterion, offsets[:nb_half*2], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        loss_emd4, _, _ = embedding_loss(emd4, down4[:,0:nb_half*1], down4[:,nb_half*1:nb_half*2], down4[:,nb_half*2:nb_half*3], criterion, offsets[:nb_half*1], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets, affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        # loss_emd1_cross, _ = ema_embedding_loss(emd1, ema_emd1, down1[:,0:nb_half*4], down1[:,nb_half*4:nb_half*8], down1[:,nb_half*8:nb_half*12], criterion, offsets[:nb_half*4], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
                                
        loss_embedding = loss_embedding * deep_weight_factor[0]
        loss_emd1 = loss_emd1 * deep_weight_factor[1]
        loss_emd2 = loss_emd2 * deep_weight_factor[2]
        loss_emd3 = loss_emd3 * deep_weight_factor[3]
        loss_emd4 = loss_emd4 * deep_weight_factor[4]

        loss_embedding_discrim = discriminative_loss(embedding, target_ins, neighbor_mask)

        loss_embedding_total = (loss_emd1 + loss_emd2 + loss_emd3 + loss_emd4 + loss_embedding) * cfg.TRAIN.self_emb
        loss_embedding_discrim_total = loss_embedding_discrim * cfg.TRAIN.dis_emb
        loss_mask = cfg.TRAIN.mask_weight * criterion_mask(pred_mask, torch.gt(target_ins[:, 0], 0), weight_rate=[10, 1]).to(device)
        
        loss = loss_embedding_total + loss_embedding_discrim_total + loss_mask
        loss.backward()
        pred = F.relu(pred)

        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        if cfg.TRAIN.lr_mode in lr_strategies:
            lr_scheduler.step()

        ### update EMA model
        if not cfg.TRAIN.sharing_weights:
            update_ema_variables(model, ema_model, cfg.TRAIN.ema_decay, iters)

        sum_loss += loss.item()
        sum_loss_embedding += loss_embedding_total.item()
        sum_loss_mask += loss_mask.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f, loss_emd=%.6f, loss_mask=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss, sum_loss_embedding, sum_loss_mask, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                logging.info('step %d, emd=%.6f, emb_discrim=%.6f' % (iters, loss_embedding_total, loss_embedding_discrim_total))
                writer.add_scalar('loss', sum_loss, iters)
            else:
                logging.info('step %d, loss=%.6f, loss_emd=%.6f, loss_mask=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                            % (iters, sum_loss / cfg.TRAIN.display_freq, \
                            sum_loss_embedding / cfg.TRAIN.display_freq, \
                            sum_loss_mask / cfg.TRAIN.display_freq, current_lr, sum_time, \
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                logging.info('step %d, emd=%.6f, emb_discrim=%.6f' % (iters, loss_embedding_total, loss_embedding_discrim_total))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss', sum_loss_embedding / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss', sum_loss_mask / cfg.TRAIN.display_freq, iters)
            # f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq))
            f_loss_txt.write('step = %d, loss = %.6f, loss_emd = %.6f, loss_mask = %.6f' % \
                (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_embedding / cfg.TRAIN.display_freq, sum_loss_mask / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0.0
            sum_loss = 0.0
            sum_loss_embedding = 0.0
            sum_loss_mask = 0.0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, batch_data['image'], pred[:,-1], batch_data['affs'][:,-1], cfg.cache_path)
            # show_affs_emb(iters, batch_data['image'], batch_data['ema_image'], pred[:,-1], batch_data['affs'][:,-1], embedding, ema_embedding, cfg.cache_path)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                dice = []
                diff = []
                all_voi = []
                all_arand = []
                all_mse = []
                all_bce = []
                for k, batch in enumerate(dataloader, 0):
                    batch_data = batch
                    inputs = batch_data['image'].cuda()
                    target = batch_data['affs'].cuda()
                    weightmap = batch_data['wmap'].cuda()
                    target_ins = batch_data['seg'].cuda()
                    affs_mask = batch_data['mask'].cuda().float()

                    with torch.no_grad():
                        emd4, emd3, emd2, emd1, embedding, pred_mask = model(inputs)

                    # tmp_loss = criterion(pred, target, weightmap)
                    loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets, affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
                    loss_mask = cfg.TRAIN.mask_weight * criterion_mask(pred_mask, torch.gt(target_ins[:, 0], 0), weight_rate=[10, 1]).to(device)
                    tmp_loss = loss_embedding + loss_mask
                    losses_valid.append(tmp_loss.item())
                    pred = F.relu(pred)
                    temp_mse = valid_mse(pred*affs_mask, target*affs_mask)
                    temp_bce = valid_bce(torch.clamp(pred, 0.0, 1.0)*affs_mask, target*affs_mask)
                    all_mse.append(temp_mse.item())
                    all_bce.append(temp_bce.item())
                    out_affs = np.squeeze(pred.data.cpu().numpy())

                    # post-processing
                    gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint8)
                    gt_mask = gt_ins.copy()
                    gt_mask[gt_mask != 0] = 1
                    pred_seg = seg_mutex(out_affs, offsets=offsets, strides=list(cfg.DATA.strides), mask=gt_mask).astype(np.uint16)
                    pred_seg = merge_func(pred_seg)
                    pred_seg = relabel(pred_seg)
                    pred_seg = pred_seg.astype(np.uint16)
                    gt_ins = gt_ins.astype(np.uint16)

                    # evaluate
                    temp_dice = BestDice(pred_seg, gt_ins)
                    temp_diff = AbsDiffFGLabels(pred_seg, gt_ins)
                    arand = adapted_rand_ref(gt_ins, pred_seg, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_ins, pred_seg, ignore_labels=(0))
                    voi_sum = voi_split + voi_merge
                    all_voi.append(voi_sum)
                    all_arand.append(arand)
                    dice.append(temp_dice)
                    diff.append(temp_diff)
                    if k == 0:
                        affs_gt = batch_data['affs'].numpy()[0,-1]
                        val_show(iters, out_affs[-1], affs_gt, pred_seg, gt_ins, cfg.valid_path)
                epoch_loss = sum(losses_valid) / len(losses_valid)
                sbd = sum(dice) / len(dice)
                # sbd = 0.0
                dic = sum(diff) / len(diff)
                mean_voi = sum(all_voi) / len(all_voi)
                mean_arand = sum(all_arand) / len(all_arand)
                mean_mse = sum(all_mse) / len(all_mse)
                mean_bce = sum(all_bce) / len(all_bce)

                # out_affs[out_affs <= 0.5] = 0
                # out_affs[out_affs > 0.5] = 1
                # whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f, MSE=%.6f, BCE=%.6f' % \
                    (iters, epoch_loss, sbd, dic, mean_voi, mean_arand, mean_mse, mean_bce), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/SBD', sbd, iters)
                writer.add_scalar('valid/DiC', dic, iters)
                writer.add_scalar('valid/VOI', mean_voi, iters)
                writer.add_scalar('valid/ARAND', mean_arand, iters)
                writer.add_scalar('valid/MSE', mean_mse, iters)
                writer.add_scalar('valid/BCE', mean_bce, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f, MSE=%.6f, BCE=%.6f' % \
                    (iters, epoch_loss, sbd, dic, mean_voi, mean_arand, mean_mse, mean_bce))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":

    # python main.py -c stmaps -m train

    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        # ema_model = build_model(cfg, writer, ema=cfg.TRAIN.ema)
        ema_model = None
        if cfg.TRAIN.opt_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.base_lr, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, ema_model, optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')