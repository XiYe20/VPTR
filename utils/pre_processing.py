from pathlib import Path
import subprocess
import shutil
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import detectron2.data.transforms as T
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch
from detectron2 import model_zoo

from itertools import groupby
from operator import itemgetter
from tqdm import tqdm
import zipfile
"""
ffmpeg command:
ffmpeg -i person01_boxing_d1_uncomp.avi -s 64x64 test%03d.png
ffmpeg -i test.avi out.mp4
"""

def unzip(file_dir, output_dir):
    with zipfile.ZipFile(file_dir, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def vid2frames(dataset_dir, root_dir, frame_size = None):
    videos_path = Path(dataset_dir)
    vid_file_list = list(videos_path.glob("*.mp4"))
    print(len(vid_file_list))
    for vid in vid_file_list:
        dir_str = ''
        for s in vid.name.strip().split('.')[:-1]:
            dir_str += s
        dir_str = dir_str.replace(" ", "")
        img_dir = Path(root_dir).joinpath(Path(dir_str))

        if img_dir.exists():
            shutil.rmtree(img_dir.absolute())
        
        img_dir.mkdir(parents=True, exist_ok=True)

        if frame_size is not None:
            command = ['ffmpeg', '-i', vid.absolute().as_posix(), '-f', 'image2', '-s', f'{frame_size}x{frame_size}', f'{img_dir.absolute().as_posix()}/image_%04d_{frame_size}x{frame_size}.png']
        else:
            #command = ['ffmpeg', '-i', vid.absolute().as_posix(), '-f', 'image2', f'{img_dir.absolute().as_posix()}/image_%04d.png']
            command = ['ffmpeg', '-i', vid.absolute().as_posix(), '-r', '10', '-s', f'64x64', f'{img_dir.absolute().as_posix()}/image_%04d.png']
        
        process = subprocess.Popen(command, stdout = subprocess.PIPE)
    
def frames2vid(frames_dir, out_file, frame_size, fps = 25):
    frames_path = Path(frames_dir)
    out_file_path = Path(out_file)
    
    command = ['ffmpeg', '-f', 'image2', '-r', f'{fps}', '-i', f'{frames_path.absolute().as_posix()}/image_%04d_{frame_size}x{frame_size}.png', 
              '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{out_file_path.absolute().as_posix()}']
    """
    command = ['ffmpeg', '-f', 'image2', '-r', f'{fps}', '-i', f'{frames_path.absolute().as_posix()}/image_%04d.png', 
              '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{out_file_path.absolute().as_posix()}']
    """
    process = subprocess.Popen(command, stdout = subprocess.PIPE)

def subsample(frames_dir, factor = 5):
    frames_path = Path(frames_dir)
    frame_list = sorted(list(frames_path.glob(f'*.png')))
    keep_list = frame_list[::factor]
    delete_list = [f for f in frame_list if f not in keep_list]
    for f in delete_list:
        f.unlink()

class detectron_detector(object):
    """
    Custom batch predicting function based on original detectron2.DefaultPredictor
    Ref: https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
    
    def __call__(self, input_img):
        """
        input_img --- numpy array with shape (H, W, C)
        """
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                input_img = input_img[:, :, ::-1]
            height, width, _ = input_img.shape

            image= self.transform_gen.get_transform(input_img).apply_image(input_img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            
            predictions = self.model([inputs])

            return predictions[0]

def human_detector(original_frames_path, save_dir, clip_length = 20):
    """
    Remove the empty KTH frames.
    original_frames_path: path for the original frames
    save_dir: save directory for the frames with human detected
    """
    all_vid_files = sorted(list(Path(original_frames_path).absolute().glob('*.avi')))
    #each person has 4 videos
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.INPUT.FORMAT = 'RGB'
    detector = detectron_detector(cfg)
    frames_folder = []

    for f in all_vid_files:
        person_id = int(str(f.name).strip().split('_')[0][-2:])
        frame_folder = f.parent.joinpath(f.name.strip().split('.')[0])
        img_files = sorted(list(frame_folder.glob('*')))
        
        scores = []
        pgbar = tqdm(total = len(img_files), desc = f'Detecting {f}')
        for img_path in img_files:
            img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            img = np.asarray(img)
            preds = detector(img)
            scores.append(preds["instances"].scores.cpu().numpy())
            #print(preds["instances"].pred_classes, preds["instances"].scores)
            pgbar.update(1)
        
        #label the frame without human as -1, frame with human as frame id
        human_frame_ids = []
        for i in range(len(scores)):
            s = scores[i]
            if len(s) > 0:
                if s[0] > 0.5: #person class id of COCO-dectection dataset is 0
                    human_frame_ids.append(i)
                else:
                    human_frame_ids.append(-1)
            else:
                human_frame_ids.append(-1)
                
        #get the subset of all conscutive frames with human detected
        consecutive_frames = []
        for k, g in groupby(enumerate(human_frame_ids), lambda x: x[0] - x[1]):
            cf = list(map(itemgetter(1), g))
            if len(cf) >= clip_length:
                consecutive_frames.append(cf)
        
        #copy all the frames with human detected to the new folder
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        for idx, cf in enumerate(consecutive_frames):
            new_folder = Path(save_dir).joinpath(frame_folder.name + f'_no_empty_{idx}')
            for f_id in cf:
                src = img_files[f_id]
                if not Path(new_folder).exists():
                    Path(new_folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(src.absolute().as_posix(), new_folder)
        

if __name__ == '__main__':
    #unzip('boxing.zip', './boxing')
    #vid2frames('/store/travail/xiyex/KTH/boxing', None)
    #frames2vid('/home/travail/xiyex/VideoFramePrediction/dataset/KTH/walking/person01_walking_d1_uncomp', './out.mp4', 64)
    """
    consecutive_frames = human_detector('/store/travail/xiyex/KTH/jogging', 
                                        '/store/travail/xiyex/KTH/jogging_no_empty')
    """
    
    vid2frames('/store/travail/xiyex/Human36Moriginal/train/S11/Videos', '/store/travail/xiyex/Human36M/train/S11')