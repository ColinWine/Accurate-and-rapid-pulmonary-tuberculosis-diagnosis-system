from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import SimpleITK as sitk
import numpy as np

def PatientData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30, resample=False, expand=False, center=None),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]),
    }
    dataset_sizes = {}
    dataloders = {}
    patient_list = []

    for patient in os.listdir('MaskVal'):
        patient_dir=os.path.join('MaskVal',patient)
        image_dataset = SliceDataSet(patient_dir,data_transforms['val'])
        # wrap your data and label into Tensor
        dataloder = torch.utils.data.DataLoader(image_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
        dataset_size = len(image_dataset)

        patient_list.append(patient)
        dataset_sizes[patient] = dataset_size
        dataloders[patient] = dataloder

    return patient_list, dataloders, dataset_sizes

class SliceDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms):
        self.img_path = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.imgs = self._make_dataset()
        self.img_min = 0
        self.img_max = 0
        self.img_range = self.img_max - self.img_min

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = np.load(data)
        img_max = np.max(img)
        img_min = np.min(img)
        img_range = img_max - img_min
        if img_range == 0:
            img = np.zeros((224, 224))
        else:
            img = (img.astype('float32') - img_min) / img_range
        PIL_image = Image.fromarray(img)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(PIL_image)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

    def _make_dataset(self):
        images = []
        if not os.path.exists(self.root_dir):
            return -1
        for filepath, dirs, names in os.walk(self.root_dir):
            if filepath.split('\\')[-1]=='1':
                abnormal = 1
            else:
                abnormal = 0
            for filename in names:
                file_path = os.path.join(filepath, filename)
                item = (file_path,abnormal)
                images.append(item)

        return images