from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

class Lymphocytes(Dataset):
    """Lymphocytes dataset."""

    def __init__(self, images_folder, annotation_file='clinical_annotation.csv', patient_bs=6, transform=None):
        """
        Args:
            images_folder (string): Path to the folder with the images.
            annotation_file (string): Path to the CSV file containing the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_folder = images_folder
        self.patient_list = os.listdir(images_folder)
        self.annotation_df = pd.read_csv(annotation_file, index_col=0)
        self.annotation_df = self.annotation_df[self.annotation_df['ID'].isin(self.patient_list)]
        self.patient_bs = patient_bs
        self.transform = transform

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        annotations = self.annotation_df.iloc[idx]
        patient_id = annotations['ID']

        path = os.path.join(self.images_folder, patient_id)
        list_imgs_paths = os.listdir(path)
        if self.patient_bs is not None:
            bs = min(self.patient_bs, len(list_imgs_paths))
            chosen_images = np.random.choice(list_imgs_paths, bs, replace=False)
        else:
            chosen_images = np.random.choice(list_imgs_paths, len(list_imgs_paths), replace=False)

        # # From numpy
        # img_array = np.array([np.array(Image.open(os.path.join(path,img_path)).convert('RGB')) for img_path in chosen_images])
        # img_tensor = torch.from_numpy(img_array).permute(0,3,1,2)

        # From torch directly
        all_imgs = []
        for img_path in chosen_images:
            im = Image.open(os.path.join(path,img_path)).convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            else:
                im = transforms.ToTensor()(im)
            all_imgs.append(im)
        img_tensor = torch.stack(all_imgs, dim=0)

        sample = {'images': img_tensor, 'annotations': annotations}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
