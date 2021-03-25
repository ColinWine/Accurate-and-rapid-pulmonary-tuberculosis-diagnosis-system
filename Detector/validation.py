import os
import sys
import cv2
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import SimpleITK as sitk
import json
from mean_average_precision.detection_map import DetectionMAP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import torch.utils.data

from nets.hourglass import get_hourglass
from utils.utils import load_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/hourglass/checkpoint.t7')
parser.add_argument('--arch', type=str, default='large_hourglass')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

def evaluation(model,validation_folder,visualization = True,map_save_name='map.png'):
  frames = []
  n_class = 1
  mAP = DetectionMAP(n_class)
  max_per_image = 100

  validation_label = r'L:\TB_Project\src\Detector\data\coco\annotations\instances_val2017.json'
  with open(validation_label,'r') as load_f:
    load_dict = json.load(load_f)
  file_dict = {}
  for file in load_dict['images']:
      file_dict[file['file_name']] = file['id']
  label_dict = {}
  for label in load_dict['annotations']:
      label_dict[label['image_id']] = []
  for label in load_dict['annotations']:
      label_dict[label['image_id']].append(label['bbox'])

  for img_file in os.listdir(validation_folder):
    img_file_path = os.path.join(validation_folder,img_file)
    image = np.load(img_file_path)

    gt_boxes = []
    for (x_0,y_0,bbox_width,bbox_height) in label_dict[file_dict[img_file]]:
      x_1 = x_0 + bbox_width
      y_1 = y_0 + bbox_height
      gt_boxes.append((x_0,y_0,x_1,y_1))
    
    height, width = image.shape[0:2]
    padding = 127 if 'hourglass' in cfg.arch else 31
    imgs = {}

    for scale in [1]:
      new_height = int(height * scale)
      new_width = int(width * scale)

      img_height = 512
      img_width = 512
      center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      scaled_size = max(height, width) * 1.0
      scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)

      img = cv2.resize(image, (new_width, new_height))
      trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
      img = cv2.warpAffine(img, trans_img, (img_width, img_height))

      img = (img.astype(np.float32)+1024.) / 4096.
      img -= 0.09102
      img /= 0.3735
      img = np.stack((img,img,img),axis=2)
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      imgs[scale] = {'image': torch.from_numpy(img).float(),
                    'center': np.array(center),
                    'scale': np.array(scaled_size),
                    'fmap_h': np.array(img_height // 4),
                    'fmap_w': np.array(img_width // 4)}

    with torch.no_grad():
      detections = []
      for scale in imgs:
        imgs[scale]['image'] = imgs[scale]['image'].cuda()

        output = model(imgs[scale]['image'])[-1]
        dets = ctdet_decode(*output, K=100) #test_topk
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

        top_preds = {}
        dets[:, :2] = transform_preds(dets[:, 0:2],
                                      imgs[scale]['center'],
                                      imgs[scale]['scale'],
                                      (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                      imgs[scale]['center'],
                                      imgs[scale]['scale'],
                                      (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
        cls = dets[:, -1]
        for j in range(1):
          inds = (cls == j)
          top_preds[j + 1] = dets[inds, :5].astype(np.float32)
          top_preds[j + 1][:, :4] /= scale

        detections.append(top_preds)

      bbox_and_scores = {}
      for j in range(1):
        temp= np.concatenate([d[j+1] for d in detections], axis=0)
        bbox_and_scores[j] = temp

      scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1)])

      if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1):
          keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
          bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

      fig = plt.figure(0)
      matplotlib.use('QT5Agg')
      plt.imshow(image)
      count = 0

      pred_bb = []
      pred_cls = []
      pred_conf = []
      gt_bb = []
      gt_cls = []
      for lab in bbox_and_scores:
        for boxes in bbox_and_scores[lab]:
          x1, y1, x2, y2, score = boxes
          if score > 0.15:
            pred_bb.append([x1, y1, x2, y2])
            pred_cls.append(0)
            pred_conf.append(score)
            count = count+1
            plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x1 + 3, y1 + 3, '%.2f' % score,
                    bbox=dict(facecolor='r', alpha=0.3), fontsize=7, color='k')
      for boxes in gt_boxes:
        x1, y1, x2, y2 = boxes
        gt_bb.append(boxes)
        gt_cls.append(0)
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=2, edgecolor='r', facecolor='none'))

      frames.append((np.array(pred_bb), np.array(pred_cls), np.array(pred_conf), np.array(gt_bb), np.array(gt_cls)))
      fig.patch.set_visible(False)
      plt.axis('off')
      if visualization:
        save_path='./result/'
        save_name=save_path+img_file[:-3]+'png'
        plt.savefig(save_name, dpi=300, transparent=True)
      plt.close('all')

  for frame in frames:
      mAP.evaluate(*frame)
  mAP.plot()
  plt.savefig(map_save_name)


def main():
  cfg.device = torch.device('cuda')
  torch.backends.cudnn.benchmark = False
  validation_folder=r'L:\TB_Project\src\Detector\data\coco\val2017'
  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  else:
    raise NotImplementedError

  model = load_model(model, cfg.ckpt_dir)
  model = model.to(cfg.device)
  model.eval()
  evaluation(model,validation_folder)

if __name__ == '__main__':
  main()
