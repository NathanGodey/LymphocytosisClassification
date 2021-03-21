from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def process_birthdate(str_date):
    """
    Args:
        str_date (string): string containing the date of birth value (eg '14-03-1999')
    """
    str_date = str_date.replace('-', '/')
    day, month, year = tuple(str_date.split('/'))
    return (int(day) + int(month) * 12 + int(year) * 365 - 693500)/766500


class Lymphocytes(Dataset):
    """
    torch.Dataset to load gray level images by batches belonging to one patient at a time
    """

    def __init__(self, images_folder, annotation_file='clinical_annotation.csv', patient_bs=6, transform=None):
        """
        Args:
            images_folder (string): Path to the folder with the images.
            annotation_file (string): Path to the CSV file containing the annotations.
            patient_bs (int): the number of images to load from each patient's bag
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_folder = images_folder
        self.patient_list = os.listdir(images_folder)
        self.annotation_df = pd.read_csv(annotation_file, index_col=0)
        self.annotation_df = self.annotation_df[self.annotation_df['ID'].isin(self.patient_list)]
        self.annotation_df = self.annotation_df.append(self.annotation_df[self.annotation_df['LABEL'] == 0])
        self.annotation_df = self.annotation_df.sample(frac=1).reset_index(drop=True)

        self.patient_bs = patient_bs
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        annotations = self.annotation_df.iloc[idx]
        patient_id = annotations['ID']

        path = os.path.join(self.images_folder, patient_id)
        list_imgs_paths = [el for el in os.listdir(path)]
        L = len(list_imgs_paths)
        list_imgs_paths = [list_imgs_paths[i%L] for i in range(self.patient_bs)]
        chosen_images = np.random.choice(list_imgs_paths, self.patient_bs, replace=False)

        # Load images as subsampled arrays from their paths
        img_array = np.array([np.array(Image.open(os.path.join(path,img_path)).convert('L'))[::4,::4]/255 for img_path in chosen_images])
        img_array = np.expand_dims(img_array, axis=3)
        img_tensor = torch.from_numpy(img_array).permute(0,3,1,2)

        sample = {'images': img_tensor.float(), 'annotations': annotations}

        if self.transform:
            sample['images'] = self.transform(sample['images'])

        return sample


class LymphocytesFlat(Dataset):
    """
    torch.Dataset to load images out of their bags with annotations
    """
    def __init__(self, images_folder, patient_list = None, annotation_file='clinical_annotation.csv'):
        """
        Args:
            images_folder (string): Path to the folder with the images.
            patient_list (list or None): list of patient IDs to use
            annotation_file (string): Path to the CSV file containing the annotations.
        """
        self.images_folder = images_folder
        if patient_list:
          self.patient_list = patient_list
        else:
          self.patient_list = os.listdir(images_folder)
        self.annotation_df = pd.read_csv(annotation_file, index_col=0)
        self.annotation_df = self.annotation_df[self.annotation_df['ID'].isin(self.patient_list)]
        self.annotation_df['DOB'] = self.annotation_df['DOB'].apply(process_birthdate)
        self.image_paths = [os.path.join(el[0], f) for el in os.walk(images_folder) for f in el[2] if os.path.isfile(os.path.join(el[0], f)) and 'bin' not in f and el[0].split('/')[-1] in self.patient_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_array = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        img_array = img_array[::4,::4]/255
        img_tensor = torch.from_numpy(img_array).permute(2,0,1)
        patient_id_str = self.image_paths[idx].split('/')[1]
        patient_id = int(patient_id_str.replace('P',''))
        label = list(self.annotation_df[self.annotation_df['ID']==patient_id_str]['LABEL'])[0]
        lymph_count = list(self.annotation_df[self.annotation_df['ID']==patient_id_str]['LYMPH_COUNT'])[0]
        dob = list(self.annotation_df[self.annotation_df['ID']==patient_id_str]['DOB'])[0]
        return img_tensor, patient_id, label, lymph_count, dob
