import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))


import torch
import torch.utils.data
import torchvision
from torch.autograd import Variable

from nets.hourglass import get_hourglass
from utils.utils import load_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctdet_decode

import se_resnet

Focus_classes=['Calcified granulomas','Cavitation','Centrilobular and tree-in-bud nodules','Clusters of nodules','Consolidation','Fibronodular scarring']

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--ckpt_detector', type=str, default='./ckpt/hourglass/checkpoint.t7')
parser.add_argument('--ckpt_classifier', type=str, default='./Classifier/epoch_45.pth.tar')
parser.add_argument('--arch', type=str, default='large_hourglass')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

def overlap_ratio(bbox1,bbox2,thresh):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # calculate area of two bbox
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # calculate area of intersection
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h
    if a1/s1 > thresh:
        return True
    if a1/s2 > thresh:
        return True
    return False

def max_bbox(bbox1,bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    return [min(xmin1, xmin2),min(ymin1, ymin2),max(xmax1, xmax2),max(ymax1, ymax2)]

def absorb(dets, thresh = 0.5):
    keep = []
    keep_conf = []
    for bbox,conf in dets:
        absorbed = 0
        for index,exist_bbox in enumerate(keep):
            if overlap_ratio(bbox,exist_bbox,thresh):
                keep[index] = max_bbox(bbox,exist_bbox)
                keep_conf[index] = max(conf,keep_conf[index])
                absorbed = 1
        if absorbed == 0:
            keep.append(bbox)
            keep_conf.append(conf)
 
    return keep,keep_conf

def EvenInt(num):
    if (int(num) % 2) == 0:
        return int(num)
    else:
        return int(num) + 1

def Resize(width,height):
    if width>height:
        return (224,EvenInt(224*height/width))
    else:
        return (EvenInt(224*width/height),224)

def Resize_Padding(in_array):
  img_max = in_array.max()
  img_min = in_array.min()
  img_range = img_max - img_min
  
  resized_shape = Resize(in_array.shape[0],in_array.shape[1])
  PIL_image = Image.fromarray(in_array)
  resized_array = torchvision.transforms.Resize(resized_shape)(PIL_image)
  (rows, cols) = resized_shape
  padding_array = np.zeros([224, 224])
  padding_array = padding_array - 1024
  Top_padding = int((224 - resized_shape[0])/2)
  Left_padding = int((224 - resized_shape[1])/2)
  padding_array[Top_padding:rows + Top_padding, Left_padding:cols + Left_padding] = resized_array
  padding_array = (padding_array.astype('float32')  - img_min)/ img_range
  out_image = Image.fromarray(padding_array)
  out_tensor = torchvision.transforms.ToTensor()(out_image)
  return out_tensor

def evaluation(model,classification_model,original_folder,target_folder,visualization = True):
  max_per_image = 100

  for img_folder in os.listdir(original_folder):
    print(img_folder)
    img_folder_path = os.path.join(original_folder,img_folder)
    for dcm_file in os.listdir(img_folder_path):
      img_path = os.path.join(img_folder_path,dcm_file)
      image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0,:,:]

      #Transform input image 
      image[np.where(image==-3024)]=-1024
      image = image.astype(np.int16)
      height, width = image.shape[0:2]
      imgs = {}
      scale = 1
      new_height = int(height * scale)
      new_width = int(width * scale)

      img_height, img_width = 512, 512
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

      #Prediction on input image
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

        #visualize predicted bounding box
        fig = plt.figure(0)
        matplotlib.use('QT5Agg')
        plt.imshow(image)

        predicts = []
        pred_bb = []
        for lab in bbox_and_scores:
          for boxes in bbox_and_scores[lab]:
            x1, y1, x2, y2, score = boxes
            if score > 0.2:
              predicts.append(([x1, y1, x2, y2],score))

        pred_bb,pred_conf = absorb(predicts) #Post-Process to merge bboxs
        for bbox in pred_bb:
          x1, y1, x2, y2 = bbox
          plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,linewidth=2, edgecolor='g', facecolor='none')) #Plot bbox

          focus_region = image[int(y1):int(y2),int(x1):int(x2)]
          out_tensor = Resize_Padding(focus_region)
          input = Variable(torch.unsqueeze(out_tensor, dim=0).float(), requires_grad=False)
          outputs = classification_model(input)
          _, preds = torch.max(outputs.data, 1) #Crop bbox region and classification

          plt.text(x1 + 3, y1 + 3, '%s' %Focus_classes[preds],bbox=dict(facecolor='g', alpha=0.3), fontsize=7, color='k') #Label predicted class

        fig.patch.set_visible(False)
        plt.axis('off')
        if visualization:
          save_path=os.path.join(target_folder,img_folder)
          if not os.path.exists(save_path):
            os.mkdir(save_path)
          save_name=os.path.join(save_path,dcm_file[:-3]+'png')
          plt.savefig(save_name, dpi=300, transparent=True) #Save visualization result
        plt.close('all')


def main():
  cfg.device = torch.device('cuda')
  torch.backends.cudnn.benchmark = False
  parent_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
  original_folder = os.path.join(parent_path,'Result\SelectedSlices') #Selected TOP10 slices directory
  target_folder = os.path.join(parent_path,'Result\DetectionResult') #Detection visualize result save path
  print('Creating model...')

  #Load Detection model
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  else:
    raise NotImplementedError

  model = load_model(model, cfg.ckpt_detector)
  model = model.to(cfg.device)
  model.eval()

  #Load classification model
  classification_model = getattr(se_resnet ,"se_resnet_18")(num_classes = 6)
  print(cfg.ckpt_classifier)
  checkpoint = torch.load(cfg.ckpt_classifier)
  base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
  classification_model.load_state_dict(base_dict)
  classification_model.train(False)

  #Run evaluation
  evaluation(model,classification_model,original_folder,target_folder)

if __name__ == '__main__':
  main()
