import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import torchvision.transforms as transforms

class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, split, im_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
    




    def load_images(self, im_path):
        r"""
        Gets all JPG images from the path specified
        and stacks them all up
        :param im_path: Path to the directory containing images
        :return: List of image file paths and corresponding labels
        """
        assert os.path.exists(im_path), "Images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
          for fname in glob.glob(os.path.join(im_path, d_name, '*.jpg')):
            ims.append(fname)
            #labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels



    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        
        # Perform center square crop of size 2448x2448
        width, height = im.size
        left = (width - 2448) // 2
        top = (height - 2448) // 2
        right = (width + 2448) // 2
        bottom = (height + 2448) // 2
        im = im.crop((left, top, right, bottom))
        
        # Resize to 64x64
        im = im.resize((64, 64), Image.ANTIALIAS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For RGB images
        ])
        im_tensor = transform(im)
        
        return im_tensor
