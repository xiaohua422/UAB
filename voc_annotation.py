import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   Modify trainval_percent to include test set
#   Modify train_percent to change train/val ratio (default 9:1)
#   
#   Test set is used as validation set in this project
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9

#-------------------------------------------------------#
#   Path to VOC dataset folder
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size", tv)
    print("train size", tr)
    
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4] + '\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    
    classes_nums = np.zeros([256], int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        
        if not os.path.exists(png_file_name):
            raise ValueError(f"Label image {png_file_name} not found. Check file existence and extension (.png).")
        
        png = np.array(Image.open(png_file_name), np.uint8)
        
        if len(np.shape(png)) > 2:
            print(f"Label image {name} has shape {np.shape(png)}, not grayscale or 8-bit color.")
            print("Label images must be grayscale or 8-bit color maps.")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print pixel value counts.")
    print('-' * 37)
    print(f"| {'Key':>15} | {'Value':>15} |")
    print('-' * 37)
    
    for i in range(256):
        if classes_nums[i] > 0:
            print(f"| {str(i):>15} | {str(classes_nums[i]):>15} |")
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected only 0 and 255 in labels. Format error.")
        print("For binary segmentation: background=0, target=1.")
        
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Detected only background pixels (0). Dataset format error.")

    print("Images in JPEGImages must be .jpg, labels in SegmentationClass must be .png.")
