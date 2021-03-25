from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import SimpleITK as sitk
import numpy as np
from skimage import measure
import torchvision
import torch
from torch.autograd import Variable
import cv2

class_dict = {'Calcified granulomas':0,'Cavitation':1,'Centrilobular and tree-in-bud nodules':2,'Clusters of nodules':3,'Consolidation':4,'Fibronodular scarring':5}

def SliceData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }
    image_datasets = {}
    full_dataset = SliceDataSet(r'L:\Classification\Classification',data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    image_datasets['train'], image_datasets['val'] = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers, 
                                                 shuffle = True) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

def read_file(file):  # 读取niifile文件
    img = sitk.GetArrayFromImage(sitk.ReadImage(file))[0,:,:]
    return img

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
    return out_image

class SliceDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms):
        self.img_path = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = sitk.GetArrayFromImage(sitk.ReadImage(data))[0,:,:]
        [dirname,filename]=os.path.split(data)
        roi_file = os.path.join(dirname,'1.nii.gz')
        merged_roi_file = os.path.join(dirname,'1_Merge.nii.gz')
        dcm = read_file(data)
        try:
            roi = read_file(roi_file)
        except:
            roi = read_file(merged_roi_file)

        y_pix,x_pix=np.where(roi==1)
        x_0=x_pix[0]
        y_0=y_pix[0]
        x_1=x_pix[-1]
        y_1=y_pix[-1]
        
        cropped_region=img[y_0:y_1,x_0:x_1]
        PIL_image = Resize_Padding(cropped_region)
        out = self.data_transforms(PIL_image)
        return out, label

    def _make_dataset(self):
        images = []
        if not os.path.exists(self.root_dir):
            return -1

        for dir, _, fnames in os.walk(self.root_dir):
            for fname in fnames:
                if fname.endswith('1.dcm'):
                    focus_class = class_dict[dir.split('\\')[-2]]
                    file_path = os.path.join(dir, fname)
                    item = (file_path,focus_class)

                    dirname = dir
                    roi_file = os.path.join(dirname,'1.nii.gz')
                    merged_roi_file = os.path.join(dirname,'1_Merge.nii.gz')
                    try:
                        roi = read_file(roi_file)
                    except:
                        roi = read_file(merged_roi_file)

                    y_pix,x_pix=np.where(roi==1)
                    if len(x_pix)>0 and len(measure.find_contours(roi,0.5))==1:
                        images.append(item)
        return images