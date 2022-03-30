from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from pathlib import Path
import shutil
from collections import OrderedDict


def resume_training(module_dict, optimizer_dict, resume_ckpt, loss_name_list = None, map_location = None):
    modules_state_dict, optimizers_state_dict, start_epoch, history_loss_dict, _ = load_ckpt(resume_ckpt, map_location)
    for k, m in module_dict.items():
        state_dict = modules_state_dict[k]
        try:
            m.load_state_dict(state_dict)
        except RuntimeError: #load the model trained by data distributed parallel
            new_state_dict = OrderedDict()
            for sk, sv in state_dict.items():
                nk = sk[7:] # remove `module.`
                new_state_dict[nk] = sv
            m.load_state_dict(new_state_dict)
    for k, m in optimizer_dict.items():
        state_dict = optimizers_state_dict[k]
        try:
            m.load_state_dict(state_dict)
        except RuntimeError:
            print('Optimizer statedict with module.')
            new_state_dict = OrderedDict()
            for sk, sv in state_dict.items():
                nk = sk[7:] # remove `module.`
                new_state_dict[nk] = sv
            m.load_state_dict(new_state_dict)

    if map_location is None:
        loss_dict = init_loss_dict(loss_name_list, history_loss_dict)
        return loss_dict, start_epoch
    else:
        return start_epoch, history_loss_dict


class AverageMeters(object):
    def __init__(self, loss_name_list):
        self.loss_name_list = loss_name_list
        self.meters = {}
        for name in loss_name_list:
            self.meters[name] = BatchAverageMeter(name, ':.10e')
    
    def iter_update(self, iter_loss_dict):
        for k, v in iter_loss_dict.items():
            self.meters[k].update(v)
    
    def epoch_update(self, loss_dict, epoch, train_flag = True):
        if train_flag:
            for k, v in loss_dict.items():
                try:
                    v.train.append(self.meters[k].avg)
                except AttributeError:
                    pass
                except KeyError:
                    v.train.append(0)
        else:
            for k, v in loss_dict.items():
                try:
                    v.val.append(self.meters[k].avg)
                except AttributeError:
                    pass
                except KeyError:
                    v.val.append(0)
        loss_dict['epochs'] = epoch

        return loss_dict

def gather_AverageMeters(aveMeter_list):
    """
    average the avg value from different rank
    Args:
        aveMeter_list: list of AverageMeters objects
    """
    AM0 = aveMeter_list[0]
    name_list = AM0.loss_name_list

    return_AM = AverageMeters(name_list)
    for name in name_list:
        avg_val = 0
        for am in aveMeter_list:
            rank_avg = am.meters[name].avg
            avg_val += rank_avg
        avg_val = avg_val/len(aveMeter_list)
        return_AM.meters[name].avg = avg_val
    
    return return_AM


class Loss_tuple(object):
    def __init__(self):
        self.train = []
        self.val = []

def init_loss_dict(loss_name_list, history_loss_dict = None):
    loss_dict = {}
    for name in loss_name_list:
        loss_dict[name] = Loss_tuple()
    loss_dict['epochs'] = 0

    if history_loss_dict is not None:
        for k, v in history_loss_dict.items():
            loss_dict[k] = v

        for k, v in loss_dict.items():
            if k not in history_loss_dict:
                lt = Loss_tuple()
                lt.train = [0] * history_loss_dict['epochs']
                lt.val = [0] * history_loss_dict['epochs']
                loss_dict[k] = lt

    return loss_dict

def write_summary(summary_writer, in_loss_dict, train_flag = True):
    loss_dict = in_loss_dict.copy()
    del loss_dict['epochs']
    if train_flag:
        for k, v in loss_dict.items():
            for i in range(len(v.train)):
                summary_writer.add_scalars(k, {'train': v.train[i]}, i+1)
    else:
        for k, v in loss_dict.items():
            for i in range(len(v.val)):
                summary_writer.add_scalars(k, {'val': v.val[i]}, i+1)

def save_ckpt(Modules_dict, Optimizers_dict, epoch, loss_dict, save_dir):
    #Save checkpoints every epoch
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True) 
    ckpt_file = Path(save_dir).joinpath(f"epoch_{epoch}.tar")
    ckpt_codes = read_code_files()

    module_state_dict = {}
    for k, m in Modules_dict.items():
        module_state_dict[k] = m.state_dict()
    optim_state_dict = {}
    for k, m in Optimizers_dict.items():
        optim_state_dict[k] = m.state_dict()
    torch.save({
        'epoch': epoch,
        'loss_dict': loss_dict, #{loss_name: [train_loss_list, val_loss_list]}
        'Module_state_dict': module_state_dict,
        'optimizer_state_dict': optim_state_dict,
        'code': ckpt_codes
    }, ckpt_file.absolute().as_posix())

def load_ckpt(ckpt_file, map_location = None):
    ckpt = torch.load(ckpt_file, map_location = map_location)

    epoch = ckpt["epoch"]
    loss_dict = ckpt["loss_dict"]
    Modules_state_dict = ckpt['Module_state_dict']
    Optimizers_state_dict = ckpt['optimizer_state_dict']
    code = ckpt['code']

    return Modules_state_dict, Optimizers_state_dict, epoch, loss_dict, code

def visualize_batch_clips(gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch, file_dir, renorm_transform = None, desc = None):
    """
        pred_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_future_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_past_frames_batch: tensor with shape (N, past_clip_length, C, H, W)
    """
    if not Path(file_dir).exists():
        Path(file_dir).mkdir(parents=True, exist_ok=True) 
    def save_clip(clip, file_name):
        imgs = []
        if renorm_transform is not None:
            clip = renorm_transform(clip)
            clip = torch.clamp(clip, min = 0., max = 1.0)
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])
    
    def append_frames(batch, max_clip_length):
        d = max_clip_length - batch.shape[1]
        batch = torch.cat([batch, batch[:, -2:-1, :, :, :].repeat(1, d, 1, 1, 1)], dim = 1)
        return batch
    max_length = max(gt_future_frames_batch.shape[1], gt_past_frames_batch.shape[1])
    if gt_past_frames_batch.shape[1] < max_length:
        gt_past_frames_batch = append_frames(gt_past_frames_batch, max_length)
    if gt_future_frames_batch.shape[1] < max_length:
        gt_future_frames_batch = append_frames(gt_future_frames_batch, max_length)
        pred_frames_batch = append_frames(pred_frames_batch, max_length)

    batch = torch.cat([gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch], dim = -1) #shape (N, clip_length, C, H, 3W)
    batch = batch.cpu()
    N = batch.shape[0]
    for n in range(N):
        clip = batch[n, ...]
        file_name = file_dir.joinpath(f'{desc}_clip_{n}.gif')
        save_clip(clip, file_name)

def read_code_files():
    """
    Read all the files under VideoFramePrediction into bytes, and return a dictionary
    key of the dict is file name (do not include root dir)
    value of the dict is bytes of each file
    """
    proj_folder = Path(__file__).resolve().parents[1].absolute()
    code_files = []
    for file in proj_folder.rglob('*'):
        file_str = str(file)
        if '.git' not in file_str and '__pycache__' not in file_str and '.ipynb_checkpoints' not in file_str:
            code_files.append(file)
    
    code_file_dict = {}
    for file_name in code_files:
        try:
            with open(file_name, 'rb') as f:
                str_name = str(file_name).strip().split('VideoFramePrediction')
                str_name = 'VideoFramePrediction' + str_name[-1]
                code_file_dict[str_name] = f.read()
        except IsADirectoryError:
            pass
    
    return code_file_dict

def write_code_files(code_file_dict, parent_dir):
    """
    Write the saved code file dictionary to disk
    parent_dir: directory to place all the saved code files
    """
    for k, v in code_file_dict.items():
        file_path = Path(parent_dir).joinpath(k)
        if not file_path.exists():
            file_path.parent.mkdir(parents = True, exist_ok=True)
        with open(file_path, 'ab') as f:
            f.write(v)

class BatchAverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L363
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def parameters_count(model):
    """
    for name, param in model.named_parameters():
        print(name, param.size())
    """
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters are {float(count)/1e6} Million")
    return count
    