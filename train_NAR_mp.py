import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR, VPTRFormerFAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import get_dataloader
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import set_seed, gather_AverageMeters

import logging
import os

import argparse

parser = argparse.ArgumentParser(description='Datadistributed training')
parser.add_argument('--port', default='12355', type=str, help = '')

def NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, save_dir, device, renorm_transform):
    VPTR_Transformer = VPTR_Transformer.eval()
    with torch.no_grad():
        past_frames, future_frames = sample
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        past_gt_feats = VPTR_Enc(past_frames)
        future_gt_feats = VPTR_Enc(future_frames)

        rec_past_frames = VPTR_Dec(past_gt_feats)
        rec_future_frames = VPTR_Dec(future_gt_feats)

        pred_future_feats = VPTR_Transformer(past_gt_feats)
        pred_future_frames = VPTR_Dec(pred_future_feats)

        N = pred_future_frames.shape[0]
        idx = min(N, 4)
        visualize_batch_clips(past_frames[0:idx, :, ...], future_frames[0:idx, :, ...], pred_future_frames[0:idx, :, ...], save_dir, renorm_transform, desc = 'pred')
        visualize_batch_clips(past_frames[0:idx, :, ...], rec_future_frames[0:idx, :, ...], rec_past_frames[0:idx, :, ...], save_dir, renorm_transform, desc = 'ae')

def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real
    
def cal_lossT(VPTR_Disc, fake_imgs, real_imgs, fake_feats, real_feats,  mse_loss, gdl_loss, bpnce, lam_pc, lam_gan):
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    #T_PC_loss = bpnce(F.normalize(real_feats, p=2.0, dim=2), F.normalize(fake_feats, p=2.0, dim=2))
    T_PC_loss = torch.zeros(1)
    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_pc * T_PC_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss# + lam_pc * T_PC_loss
    
    return loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan


def init_models(img_channels, encC, encH, encW, dropout, out_layer, rpe, TSLMA_flag, rank, batch_size, world_size,
                Transformer_lr, resume_AE_ckpt, 
                resume_Transformer_ckpt = None, num_encoder_layers = 4, num_decoder_layers = 8,
                num_past_frames = 10, num_future_frames = 10, init_Disc = False, train_Disc = False):
    
    VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 3).to(rank)
    VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 3, out_layer = out_layer).to(rank)
    #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    start_epoch, history_loss_dict = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, map_location = f'cuda:{rank}')
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'T_bpc', 'Dtotal', 'Dfake', 'Dreal']
    loss_dict = init_loss_dict(loss_name_list, history_loss_dict)

    VPTR_Enc = DDP(VPTR_Enc, device_ids=[rank])
    VPTR_Dec = DDP(VPTR_Dec, device_ids=[rank])
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    if init_Disc:
        VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(rank)
        init_weights(VPTR_Disc)
        if not train_Disc:
            _, _ = resume_training({'VPTR_Disc': VPTR_Disc}, {}, resume_AE_ckpt, map_location = f'cuda:{rank}')
        VPTR_Disc = DDP(VPTR_Disc, device_ids=[rank])
        if not train_Disc:
            VPTR_Disc = VPTR_Disc.eval()

    VPTR_Transformer = VPTRFormerNAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                        nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout, 
                                        window_size=4, Spatial_FFN_hidden_ratio=4, TSLMA_flag = TSLMA_flag, rpe = rpe).to(rank)
    
    optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

    if resume_Transformer_ckpt is not None:
        start_epoch, history_loss_dict = resume_training({'VPTR_Transformer': VPTR_Transformer}, {'optimizer_T':optimizer_T}, resume_Transformer_ckpt, map_location = f'cuda:{rank}')
        loss_dict = init_loss_dict(loss_name_list, history_loss_dict)
    VPTR_Transformer = DDP(VPTR_Transformer, device_ids=[rank])

    optimizer_D = None
    gan_loss = None
    if train_Disc:
        optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
        gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(rank)
    
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)
    bpnce = BiPatchNCE(batch_size//world_size, num_future_frames, encH, encW, 1.0).to(rank)

    return VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_D, optimizer_T, start_epoch, loss_dict, mse_loss, gdl_loss, bpnce, gan_loss, loss_name_list

def single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, rank, mse_loss, gdl_loss, bpnce, lam_pc, gan_loss, lam_gan, max_grad_norm, train_flag = True):
    past_frames, future_frames = sample
    past_frames = past_frames.to(rank)
    future_frames = future_frames.to(rank)
    
    with torch.no_grad():
        past_gt_feats = VPTR_Enc(past_frames)
        future_gt_feats = VPTR_Enc(future_frames)
        
    if train_flag:
        VPTR_Transformer = VPTR_Transformer.train()
        VPTR_Transformer.zero_grad(set_to_none=True)
        VPTR_Dec.zero_grad(set_to_none=True)
        
        pred_future_feats = VPTR_Transformer(past_gt_feats)
        pred_frames = VPTR_Dec(pred_future_feats)
        
        if optimizer_D is not None:
            assert lam_gan is not None, "Input lam_gan"
            #update discriminator
            VPTR_Disc = VPTR_Disc.train()
            for p in VPTR_Disc.parameters():
                p.requires_grad_(True)
            VPTR_Disc.zero_grad(set_to_none=True)
            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            loss_D.backward()
            optimizer_D.step()
        
            for p in VPTR_Disc.parameters():
                    p.requires_grad_(False)
        
        pred_future_feats = VPTR_Transformer.module.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        future_gt_feats = VPTR_Transformer.module.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        #update Transformer (generator)
        loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames, pred_future_feats, future_gt_feats,  mse_loss, gdl_loss, bpnce, lam_pc, lam_gan)
        loss_T.backward()
        nn.utils.clip_grad_norm_(VPTR_Transformer.parameters(), max_norm=max_grad_norm, norm_type=2)
        optimizer_T.step()

    else:
        if optimizer_D is not None:
            VPTR_Disc = VPTR_Disc.eval()
        VPTR_Transformer = VPTR_Transformer.eval()
        with torch.no_grad():
            pred_future_feats = VPTR_Transformer(past_gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
            if optimizer_D is not None:
                loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            pred_future_feats = VPTR_Transformer.module.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            future_gt_feats = VPTR_Transformer.module.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames, pred_future_feats, future_gt_feats,  mse_loss, gdl_loss, bpnce, lam_pc, lam_gan)
        
    if optimizer_D is None:
        loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    iter_loss_dict = {'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_gan': loss_T_gan.item(), 'T_GDL': T_GDL_loss.item(),  'T_bpc':T_PC_loss.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
    
    return iter_loss_dict

def setup(rank, world_size, args):
    # initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, args, world_size, img_channels, encC, encH, encW, dropout, out_layer, TSLMA_flag,
                rpe, Transformer_lr, max_grad_norm, lam_pc, lam_gan, resume_AE_ckpt,
                data_set_name, batch_size, data_set_dir, dev_set_size, epochs, ckpt_save_dir, tensorboard_save_dir,
                resume_Transformer_ckpt = None, num_encoder_layers = 4, num_decoder_layers = 8, num_past_frames = 10, 
                num_future_frames = 10, init_Disc = False, train_Disc = False,
                num_workers = 8, show_example_epochs = 10, save_ckpt_epochs = 2):
    setup(rank, world_size, args)
    torch.cuda.set_device(rank)

    if rank == 0:
        #############Set the logger#########
        if not Path(ckpt_save_dir).exists():
                Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        format='%(asctime)s - %(message)s',
                        filename=ckpt_save_dir.joinpath('train_log.log').absolute().as_posix(),
                        filemode='w')
        summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    
    VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, \
    optimizer_D, optimizer_T, start_epoch, loss_dict, \
    mse_loss, gdl_loss, bpnce, gan_loss, loss_name_list = init_models(img_channels, encC, encH, encW, dropout, out_layer, rpe, 
                                                                      TSLMA_flag, rank, batch_size, world_size, Transformer_lr, resume_AE_ckpt, 
                                                                      resume_Transformer_ckpt, num_encoder_layers, num_decoder_layers,
                                                                      num_past_frames, num_future_frames, init_Disc, train_Disc)
    train_loader, val_loader, _, renorm_transform = get_dataloader(data_set_name, batch_size, data_set_dir, ngpus = world_size, num_workers = num_workers)
    
    for epoch in range(start_epoch+1, start_epoch + epochs+1):
        epoch_st = datetime.now()

        #Train
        print(f'train rank {rank} epoch {epoch}')
        train_EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(train_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, 
                                         sample, rank, mse_loss, gdl_loss, bpnce, lam_pc, gan_loss, lam_gan, max_grad_norm, train_flag = True)
            train_EpochAveMeter.iter_update(iter_loss_dict)
        
        print(f'rank{rank}, finish train')
        train_ave_meters = [None for i in range(world_size)]
        dist.all_gather_object(train_ave_meters, train_EpochAveMeter)
        print(f'rank{rank}, gather')
        if rank == 0:
            train_meter = gather_AverageMeters(train_ave_meters)
            loss_dict = train_meter.epoch_update(loss_dict, epoch, train_flag = True)
            write_summary(summary_writer, loss_dict, train_flag = True)
            if epoch % show_example_epochs == 0 or epoch == 1:
                NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, ckpt_save_dir.joinpath(f'train_gifs_epoch{epoch}'), rank, renorm_transform)
        
        print(f'valid rank {rank} epoch {epoch}')
        #validation
        val_EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(val_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, 
                                         sample, rank, mse_loss, gdl_loss, bpnce, lam_pc, gan_loss, lam_gan, max_grad_norm, train_flag = False)
            val_EpochAveMeter.iter_update(iter_loss_dict)
        val_ave_meters = [None for i in range(world_size)]
        print(f'valid rank{rank}, gather')
        dist.all_gather_object(val_ave_meters, val_EpochAveMeter)
        print(f'valid rank{rank}, finish gather')
        if rank == 0:
            val_meter = gather_AverageMeters(val_ave_meters)
            loss_dict = val_meter.epoch_update(loss_dict, epoch, train_flag = False)
            write_summary(summary_writer, loss_dict, train_flag = False)
            if epoch % show_example_epochs == 0 or epoch == 1:
                NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, ckpt_save_dir.joinpath(f'test_gifs_epoch{epoch}'), rank, renorm_transform)
            
            if epoch % save_ckpt_epochs == 0:
                save_ckpt({'VPTR_Transformer': VPTR_Transformer}, 
                        {'optimizer_T': optimizer_T}, epoch, loss_dict, ckpt_save_dir)
            epoch_time = datetime.now() - epoch_st
            logging.info(f"epoch {epoch}, {val_meter.meters['T_total'].avg}")
            logging.info(f"Estimated remaining training time: {epoch_time.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")
        print(f'rank {rank} epoch {epoch}')
    cleanup()

if __name__ == '__main__':
    set_seed(3407)
    args = parser.parse_args()

    ckpt_save_dir = Path('/home/travail/xiyex/VPTR_ckpts/MNIST_TLMA_aug_BPNCE01_MSEGDL_NAR_mp_ckpt')
    tensorboard_save_dir = Path('/home/travail/xiyex/VPTR_ckpts/MNIST_TLMA_aug_BPNCE01_MSEGDL_NAR_mp_tensorboard')
    resume_AE_ckpt = Path('/home/travail/xiyex/VPTR_ckpts/MNIST_ResNetAE_MSEGDLgan001_ckpt').joinpath('epoch_93.tar')
    #resume_Transformer_ckpt = ckpt_save_dir.joinpath('epoch_90.tar')
    resume_Transformer_ckpt = None

    data_set_name = 'MNIST'
    out_layer = 'Sigmoid'
    data_set_dir = '/home/travail/xiyex/MovingMNIST'
    dev_set_size = 50

    num_past_frames = 10
    num_future_frames = 10
    encH, encW, encC = 8, 8, 528
    img_channels = 1
    epochs = 150
    batch_size = 2
    num_encoder_layers = 4
    num_decoder_layers = 8

    Transformer_lr = 1e-4
    max_grad_norm = 1.0 
    TSLMA_flag = False
    rpe = False

    lam_gan = None #0.001
    lam_pc = 0.1
    dropout = 0.1

    init_Disc = False
    train_Disc = False
    num_workers = 1
    world_size = 4

    show_example_epochs = 1
    save_ckpt_epochs = 1

    print("Start training....")
    mp.spawn(main_worker,
             args=(args, world_size, img_channels, encC, encH, encW, dropout, out_layer, TSLMA_flag,
                rpe, Transformer_lr, max_grad_norm, lam_pc, lam_gan, resume_AE_ckpt,
                data_set_name, batch_size, data_set_dir, dev_set_size, epochs, ckpt_save_dir, tensorboard_save_dir,
                resume_Transformer_ckpt, num_encoder_layers, num_decoder_layers, num_past_frames, 
                num_future_frames, init_Disc, train_Disc,
                num_workers, show_example_epochs, save_ckpt_epochs),
             nprocs=world_size)
    