import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset, write_code_files
from utils import VidCenterCrop, VidPad, VidResize, VidNormalize, VidReNormalize, VidCrop, VidRandomHorizontalFlip, VidRandomVerticalFlip, VidToTensor
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import set_seed, get_dataloader

import logging

def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real
    
def cal_lossT(VPTR_Disc, fake_imgs, real_imgs, fake_feats, real_feats, lam_pc, lam_gan):
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    T_PC_loss = bpnce(F.normalize(real_feats, p=2.0, dim=2), F.normalize(fake_feats, p=2.0, dim=2))

    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_pc * T_PC_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss + lam_pc * T_PC_loss
    
    return loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan

def single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, train_flag = True):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    
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
        
            #update Transformer (generator)
            for p in VPTR_Disc.parameters():
                p.requires_grad_(False)

        pred_future_feats = VPTR_Transformer.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        future_gt_feats = VPTR_Transformer.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames, pred_future_feats, future_gt_feats, lam_pc, lam_gan)
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

            pred_future_feats = VPTR_Transformer.NCE_projector(pred_future_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            future_gt_feats = VPTR_Transformer.NCE_projector(future_gt_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            loss_T, T_GDL_loss, T_MSE_loss, T_PC_loss, loss_T_gan = cal_lossT(VPTR_Disc, pred_frames, future_frames, pred_future_feats, future_gt_feats, lam_pc, lam_gan)
    
    if optimizer_D is None:        
        loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    
    iter_loss_dict = {'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_gan': loss_T_gan.item(), 'T_GDL': T_GDL_loss.item(),  'T_bpc':T_PC_loss.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
    
    return iter_loss_dict

def NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, save_dir):
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

        TP = past_frames.shape[1]
        TF = future_frames.shape[1]
        if TP < TF:
            N, _, C, H, W = past_frames.shape
            past_frames = torch.cat([past_frames, torch.zeros(N, TF-TP, C, H, W).to(past_frames.device)], dim = 1)
            rec_past_frames = torch.cat([rec_past_frames, torch.zeros(N, TF-TP, C, H, W).to(rec_past_frames.device)], dim = 1)

        visualize_batch_clips(past_frames[0:idx, :, ...], future_frames[0:idx, :, ...], pred_future_frames[0:idx, :, ...], save_dir, renorm_transform, desc = 'pred')
        visualize_batch_clips(past_frames[0:idx, :, ...], rec_future_frames[0:idx, :, ...], rec_past_frames[0:idx, :, ...], save_dir, renorm_transform, desc = 'ae')


if __name__ == '__main__':
    set_seed(2021)

    ckpt_save_dir = Path('/home/travail/xiyex/VPTR_ckpts/BAIR_NAR_MSEGDL_BPNCE01_RPE_ckpt')
    tensorboard_save_dir = Path('/home/travail/xiyex/VPTR_ckpts/BAIR_NAR_MSEGDL_BPNCE01_RPE_tensorboard')
    resume_AE_ckpt = Path('/home/travail/xiyex/VPTR_ckpts/BAIR_ResNetAE_MSEGDL_ckpt').joinpath('epoch_64.tar')
    #resume_ckpt = ckpt_save_dir.joinpath('epoch_88.tar')
    resume_ckpt = None

    #############Set the logger#########
    if not Path(ckpt_save_dir).exists():
            Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    format='%(asctime)s - %(message)s',
                    filename=ckpt_save_dir.joinpath('train_log.log').absolute().as_posix(),
                    filemode='a')

    start_epoch = 0

    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    num_past_frames = 2
    num_future_frames = 10
    encH, encW, encC = 8, 8, 528
    img_channels = 3
    epochs = 100
    N = 16
    #AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0 
    TSLMA_flag = False
    rpe = True
    padding_type = 'zero'

    lam_gan = None #0.001
    lam_pc = 0.1
    device = torch.device('cuda:0')

    show_example_epochs = 10
    save_ckpt_epochs = 2

    #####################Init Dataset ###########################
    data_set_name = 'BAIR'
    dataset_dir = '/home/travail/xiyex/BAIR'
    test_past_frames = 2
    test_future_frames = 10
    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, test_past_frames, test_future_frames)

    #####################Init model###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 3, padding_type = padding_type).to(device)
    VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh', padding_type = padding_type).to(device)
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    #VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    #VPTR_Disc = VPTR_Disc.eval()
    #init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    VPTR_Transformer = VPTRFormerNAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                    nhead=8, num_encoder_layers=4, num_decoder_layers=8, dropout=0.1, 
                                    window_size=4, Spatial_FFN_hidden_ratio=4, TSLMA_flag = TSLMA_flag, rpe = rpe).to(device)
    optimizer_D = None
    #optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
    optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

    Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)
    print(f"NAR Transformer num_parameters: {Transformer_parameters}")

    #####################Init loss function###########################
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'T_bpc', 'Dtotal', 'Dfake', 'Dreal']
    #gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    bpnce = BiPatchNCE(N, num_future_frames, 8, 8, 1.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)

    #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list)

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
                                                {'optimizer_T':optimizer_T}, resume_ckpt, loss_name_list)
    
    #####################Train ################################
    for epoch in range(start_epoch+1, start_epoch + epochs+1):
        epoch_st = datetime.now()
        
        #Train
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(train_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, train_flag = True)
            EpochAveMeter.iter_update(iter_loss_dict)
            
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = True)
        write_summary(summary_writer, loss_dict, train_flag = True)

        if epoch % show_example_epochs == 0 or epoch == 1:
            NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, ckpt_save_dir.joinpath(f'train_gifs_epoch{epoch}'))
                
        #validation
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(val_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, train_flag = False)
            EpochAveMeter.iter_update(iter_loss_dict)
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = False)
        write_summary(summary_writer, loss_dict, train_flag = False)

        if epoch % save_ckpt_epochs == 0:
            save_ckpt({'VPTR_Transformer': VPTR_Transformer}, 
                    {'optimizer_T': optimizer_T}, 
                    epoch, loss_dict, ckpt_save_dir)
        
        if epoch % show_example_epochs == 0 or epoch == 1:
            for idx, sample in enumerate(test_loader, random.randint(0, len(test_loader) - 1)):
                NAR_show_samples(VPTR_Enc, VPTR_Dec, VPTR_Transformer, sample, ckpt_save_dir.joinpath(f'test_gifs_epoch{epoch}'))
                break
            
        epoch_time = datetime.now() - epoch_st

        logging.info(f"epoch {epoch}, {EpochAveMeter.meters['T_total']}")
        logging.info(f"Estimated remaining training time: {epoch_time.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")