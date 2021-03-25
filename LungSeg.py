import SimpleITK as sitk
import numpy as np
import pydicom
import os
from lungmask import mask
from lungmask import utils


parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
dataset_folder = os.path.join(parent_path,'Dataset')

def get_dcm_file_path(work_path):
    allfnames = []
    for dir, _, fnames in os.walk(work_path):
        for fname in fnames:
            if fname.endswith('.dcm'):
                allfnames.append(os.path.join(dir, fname))
            elif fname.endswith('.gz'):
                os.remove(os.path.join(dir,fname))
            else:
                new_name = fname + '.dcm'
                os.rename(os.path.join(dir, fname), os.path.join(dir, new_name))
                allfnames.append(os.path.join(dir, new_name))

    return allfnames

def LungSeg(filepaths):
    for dcm_file in filepaths:
        input_image = utils.get_input_image(dcm_file)
        segmentation = mask.apply_fused(input_image)
        [dirname,filename]=os.path.split(dcm_file)
        fname,ext = os.path.splitext(filename)
        save_file_name = os.path.join(dirname,'lobes_'+fname+'.nii.gz')
        segimg = sitk.GetImageFromArray(segmentation)
        sitk.WriteImage(segimg,save_file_name)


def PreprocessMaskNii():
    dcm_list = get_dcm_file_path(dataset_folder)
    LungSeg(dcm_list)



if __name__ == '__main__':
    PreprocessMaskNii()