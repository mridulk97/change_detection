import os
import pickle
import random

import kornia as K
import numpy as np
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torch

import utils.general
import utils.geometry
from data.augmentation import AugmentationPipeline
from utils.general import cache_data_triton



class SecondDataset(Dataset):
    def __init__(self, path_to_dataset, split, method, image_size, image_transformation=None, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indicies = train_val_test_split[split]
        self.image_name = train_val_test_split[split]
        self.split = split
        self.machine = machine
        # self.inpainted_image_names = self.get_inpainted_image_names()
        self.image_augmentations = AugmentationPipeline(
            mode=split, path_to_dataset=path_to_dataset, image_transformation=image_transformation
        )
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.image_size = image_size

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        elif method == "segmentation":
            from models.segmentation_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_train_val_test_split(self, split):
        train_val_test_split_file_path = os.path.join(self.path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)
        
        with open(os.path.join(self.path_to_dataset, "image_full_paths.txt")) as f:
            image_filename = f.readlines()
        image_filename = [x[:-1] for x in image_filename]
        np.random.shuffle(image_filename)
        if split == "test":
            train_val_test_split = {
                "test": image_filename,
            }
        else:
            number_of_images = len(image_filename)
            number_of_train_images = int(0.9 * number_of_images)
            train_val_test_split = {
                "train": image_filename[:number_of_train_images],
                "val": image_filename[number_of_train_images:],
            }
        with open(train_val_test_split_file_path, "wb") as file:
            pickle.dump(train_val_test_split, file)
        return train_val_test_split

    def get_inpainted_image_names(self):
        filenames_as_list = list(os.listdir(os.path.join(self.path_to_dataset, "inpainted")))
        inpainted_image_names = dict()
        for filename in filenames_as_list:
            index = int(filename.split("_")[0])
            if index in inpainted_image_names.keys():
                inpainted_image_names[index].append(filename)
            else:
                inpainted_image_names[index] = [filename]
        return inpainted_image_names

    def read_image_as_tensor(self, path_to_image):
        """
        Returns a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def read_bitmask_as_tensor(self, path_to_image):
        """
        Returns a segmentation bitmask image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("L")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        pil_image = pil_image.point(lambda x: 0 if x<255 else 255, '1')
        image_as_tensor = pil_to_tensor(pil_image).float()
        return image_as_tensor


    def get_inpainted_objects_bitmap_from_image_path(self, image_path, bit_length):
        return NotImplementedError

    def add_random_objects(self, image_as_tensor, item_index):
        all_indices_except_current = list(range(item_index)) + list(
            range(item_index + 1, len(self.indicies))
        )
        random_image_index = random.choice(all_indices_except_current)
        index = self.indicies[random_image_index]
        original_image = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, f"images_and_masks/{index}.png", self.machine)
        )
        annotation_path = cache_data_triton(
            self.path_to_dataset, f"metadata/{index}.npy", self.machine
        )
        annotations = np.load(annotation_path, allow_pickle=True)
        (
            original_image_resized_to_current,
            annotations_resized,
        ) = utils.geometry.resize_image_and_annotations(
            original_image, image_as_tensor.shape[-2:], annotations
        )
        annotation_mask = utils.general.coco_annotations_to_mask_np_array(
            annotations_resized, image_as_tensor.shape[-2:]
        )
        image_as_tensor = rearrange(image_as_tensor, "c h w -> h w c")
        original_image_resized_to_current = rearrange(
            original_image_resized_to_current, "c h w -> h w c"
        )
        image_as_tensor[annotation_mask] = original_image_resized_to_current[annotation_mask]
        return rearrange(image_as_tensor, "h w c -> c h w"), annotations_resized

    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indicies)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image_file_name = self.indicies[item_index]
        image_1 = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('im1', image_file_name), self.machine))
        image_2 = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('im2', image_file_name), self.machine))
        label_1 = self.read_bitmask_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('label1', image_file_name), self.machine))
        label_2 = self.read_bitmask_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('label2', image_file_name), self.machine))
                
        (
            image_1,
            image_2,
            label_1,
            label_2,
        ) = self.image_augmentations(
            image_1,
            image_2,
            label_1,
            label_2,
            item_index,
        )

        return {
            "image1": image_1.squeeze(),
            "image2": image_2.squeeze(),
            "label1": label_1.squeeze(),
            "label2": label_2.squeeze(),
        }

class SecondDataset_test_as_val(Dataset):
    def __init__(self, path_to_dataset, split, method, image_size, image_transformation=None, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indicies = train_val_test_split[split]
        self.image_name = train_val_test_split[split]
        self.split = split
        self.machine = machine
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.image_size = image_size

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        elif method == "segmentation":
            from models.segmentation_with_coam import marshal_getitem_data
        else:
            from models.centernet_with_coam import marshal_getitem_data
        return marshal_getitem_data

    def get_train_val_test_split(self, split):
        with open(os.path.join(self.path_to_dataset, "image_full_paths.txt")) as f:
            image_filename = f.readlines()
        train_image_filename = [x[:-1] for x in image_filename]
        self.path_to_test_dataset = '/'.join(self.path_to_dataset.split('/')[:-1])
        self.path_to_test_dataset = os.path.join(self.path_to_test_dataset, 'test')
        with open(os.path.join(self.path_to_test_dataset, "image_full_paths.txt")) as f:
            test_image_filename = f.readlines()
        test_image_filename = [x[:-1] for x in test_image_filename]
        np.random.shuffle(image_filename)
        if split == "test":
            train_val_test_split = {
                "test": image_filename,
            }
        else:
            train_val_test_split = {
                "train": train_image_filename,
                "val": test_image_filename,
            }
        return train_val_test_split


    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def read_bitmask_as_tensor(self, path_to_image):
        """
        Returms a segmentation bitmask image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("L")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        pil_image = pil_image.point(lambda x: 0 if x<255 else 255, '1')
        image_as_tensor = pil_to_tensor(pil_image).float()
        # image_as_tensor = pil_to_tensor(pil_image).to(torch.int64)
        return image_as_tensor


    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indicies)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image_file_name = self.indicies[item_index]
        image_1 = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('im1', image_file_name), self.machine))
        image_2 = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('im2', image_file_name), self.machine))
        label_1 = self.read_bitmask_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('label1', image_file_name), self.machine))
        label_2 = self.read_bitmask_as_tensor(
                cache_data_triton(self.path_to_dataset, os.path.join('label2', image_file_name), self.machine))

        return {
            "image1": image_1.squeeze(),
            "image2": image_2.squeeze(),
            "label1": label_1.squeeze(),
            "label2": label_2.squeeze(),
        }

