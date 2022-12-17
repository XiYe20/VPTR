import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import get_dataloader
from utils import VidCenterCrop, VidPad, VidResize, VidNormalize, VidReNormalize, VidCrop, VidRandomHorizontalFlip, VidRandomVerticalFlip, VidToTensor
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training, parameters_count
from utils import set_seed

from customDataset import rbc_data

set_seed(2021)

from train_AutoEncoder import *

import argparse
import os

def arg_def():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type = str, default = '/Users/combi/Documents/Projects/MG-Turbulent-Flow/data')
    parser.add_argument("--checkpoint_dir", type = str, default = "VPTR_ckpts")
    parser.add_argument("--batch_size",     type = int, default = 32)
    
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    return args

if __name__ == '__main__':
    args = arg_def()
    
    checkpoints_save_dir = Path(args.checkpoint_dir) / 'RBC_ResNetAE_MSEGDLgan_ckpt'
    tensorboard_save_dir = Path(args.checkpoint_dir) / 'RBC_ResNetAE_MSEGDLgan_tensorboard'

    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    
    start_epoch = 0
    num_past_frames = 16
    num_future_frames = 4
    encH, encW, encC = 8, 8, 528
    img_channels = 2
    epochs = 50
    N = 32
    AE_lr = 2e-4
    lam_gan = 0.01

    #####################Init Dataset ###########################
    data_dir  = Path(args.data_dir)
    data_prep = [torch.load(data_dir / f'sample_{i}.pt') for i in range(7)]
    
    train_set = rbc_data(data_prep, list(range(7000)), 16, 4, False)
    valid_set = rbc_data(data_prep, list(range(7000, 7500)), 16, 4, False)
    test_set = rbc_data(data_prep, list(range(7500, 7600)), 16, 4, False)
    
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=1, drop_last = True)
    valid_loader = DataLoader(valid_set, batch_size=N, shuffle=True, num_workers=1, drop_last = True)
    test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=1, drop_last = True)
    #####################Init Models and Optimizer ###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim = encC, n_downsampling = 2).to(args.device)
    VPTR_Dec = VPTRDec(img_channels, feat_dim = encC, n_downsampling = 2, out_layer = 'Tanh').to(args.device) #Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Disc = VPTRDisc(img_channels, ndf = 64, n_layers = 3, norm_layer = nn.BatchNorm2d).to(args.device)
    init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    optimizer_G = torch.optim.Adam(params = list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()), lr=AE_lr, betas = (0.5, 0.999))
    optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr=AE_lr, betas = (0.5, 0.999))
    
    print(f"Encoder num_parameters: {parameters_count(VPTR_Enc)}")
    print(f"Decoder num_parameters: {parameters_count(VPTR_Dec)}")
    print(f"Discriminator num_parameters: {parameters_count(VPTR_Disc)}")

    #####################Init Criterion ###########################
    loss_name_list = ['AE_MSE', 'AE_GDL', 'AE_total', 'Dtotal', 'Dfake', 'Dreal', 'AEgan']
    gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(args.device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)
    
    if os.path.exists(checkpoints_save_dir):
        loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec, 'VPTR_Disc': VPTR_Disc}, 
                                                 {'optimizer_G': optimizer_G, 'optimizer_D': optimizer_D}, 
                                                 resume_ckpt, loss_name_list, map_location = args.device)

    #####################Training loop ###########################                                            
    for epoch in range(start_epoch+1, start_epoch + epochs+1):
        epoch_st = datetime.now()
        
        #Train
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(train_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, optimizer_G, optimizer_D, sample, args.device, train_flag = True)
            EpochAveMeter.iter_update(iter_loss_dict)
            
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = True)
        write_summary(summary_writer, loss_dict, train_flag = True)
        
        #validation
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(val_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, optimizer_G, optimizer_D, sample, args.device, train_flag = False)
            EpochAveMeter.iter_update(iter_loss_dict)
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = False)
        write_summary(summary_writer, loss_dict, train_flag = False)
        
        save_ckpt({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec, 'VPTR_Disc': VPTR_Disc}, 
                {'optimizer_G': optimizer_G, 'optimizer_D': optimizer_D}, 
                epoch, loss_dict, ckpt_save_dir)
            
        epoch_time = datetime.now() - epoch_st
        print(f'epoch {epoch}', EpochAveMeter.meters['AE_total'])
        print(f"Estimated remaining training time: {epoch_time.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours")
        