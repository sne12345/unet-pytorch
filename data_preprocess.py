import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

def load_tif(dir_data):

    name_label = 'train-labels.tif'
    name_input = 'train-volume.tif'

    img_label = Image.open(os.path.join(dir_data, name_label))
    img_input = Image.open(os.path.join(dir_data, name_input))

    ny, nx = img_label.size
    nframe = img_label.n_frames

    return img_input, img_label, ny, nx, nframe

def make_dir(dir_data):

    dir_save_train = os.path.join(dir_data, 'train')
    dir_save_val = os.path.join(dir_data, 'val')
    dir_save_test = os.path.join(dir_data, 'test')

    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)
    if not os.path.exists(dir_save_val):
        os.makedirs(dir_save_val)
    if not os.path.exists(dir_save_test):
        os.makedirs(dir_save_test)

    return dir_save_train, dir_save_val, dir_save_test

    
def split_data(nframe_train, nframe_val, nframe_test):

    id_frame = np.arange(nframe)
    np.random.shuffle(id_frame)
    print(id_frame)

    # train
    offset = 0

    for i in range(nframe_train):
        img_label.seek(id_frame[i + offset]) # 해당 이미지 위치로 이동
        img_input.seek(id_frame[i + offset])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_train, 'label_%03d.npy'%i), label_)
        np.save(os.path.join(dir_save_train, 'input_%03d.npy'%i), input_)
        
    # val
    offset += nframe_train
    
    for i in range(nframe_val):
        img_label.seek(id_frame[i + offset]) # 해당 이미지 위치로 이동
        img_input.seek(id_frame[i + offset])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_val, 'label_%03d.npy'%i), label_)
        np.save(os.path.join(dir_save_val, 'input_%03d.npy'%i), input_)
        
    # test
    offset += nframe_val
    
    for i in range(nframe_test):
        img_label.seek(id_frame[i + offset]) # 해당 이미지 위치로 이동
        img_input.seek(id_frame[i + offset])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_test, 'label_%03d.npy'%i), label_)
        np.save(os.path.join(dir_save_test, 'input_%03d.npy'%i), input_)
        

    # save fig
    plt.subplot(121)
    plt.imshow(label_, cmap='gray')
    plt.title('label')

    plt.subplot(122)
    plt.imshow(input_, cmap='gray')
    plt.title('input')

    plt.savefig('./data/sample.png')


if __name__ == '__main__':

    dir_data = './data'

    nframe_train = 24
    nframe_val = 3
    nframe_test = 3

    img_input, img_label, ny, nx, nframe = load_tif(dir_data)
    dir_save_train, dir_save_val, dir_save_test = make_dir(dir_data)
    split_data(nframe_train, nframe_val, nframe_test)