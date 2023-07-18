import os
from argparse import ArgumentParser
import torch

import matplotlib.pyplot as plt
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from data.datamodule import DataModule
from models.segmentation_with_coam import SegmentationWithCoAttention
from utils.general import get_easy_dict_from_yaml_file


class SinglePair(Dataset):
    def __init__(self, method, file_path, image_size):
        self.file_path = file_path
        self.left_image = os.path.join(self.file_path,'before.png')
        self.right_image = os.path.join(self.file_path,'after.png')
        self.left_image_mask = os.path.join(self.file_path,'before_label.png')
        self.right_image_mask = os.path.join(self.file_path,'after_label.png')
        self.split = "test"
        self.image_size = image_size
        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.segmentation_with_coam import marshal_getitem_data
        if method == "segmentation":
            from models.segmentation_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
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
        return image_as_tensor

    def __len__(self):
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image1 = self.read_image_as_tensor(self.left_image)
        image2 = self.read_image_as_tensor(self.right_image)
        label1 = self.read_bitmask_as_tensor(self.left_image_mask)
        label2 = self.read_bitmask_as_tensor(self.right_image_mask)
        return {
            "image1": image1,
            "image2": image2,
            "label1": label1,
            "label2": label2,
        }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = SegmentationWithCoAttention.add_model_specific_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--file_path", required=True)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]
            
    image_size = configs['datasets']['train_dataset']['args']['image_size']
    dataset = SinglePair(method="segmentation", file_path=args.file_path, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = SegmentationWithCoAttention(configs)
    model.eval()

    for batch_with_single_item in dataloader:
        left_image_outputs, right_image_outputs = model.predict(
            batch_with_single_item, 0)
        left_pred_bit_mask = torch.argmax(left_image_outputs, dim=1)
        right_pred_bit_mask = torch.argmax(right_image_outputs, dim=1)
        left_pred_bit_mask = left_pred_bit_mask[0].detach().numpy()
        right_pred_bit_mask = right_pred_bit_mask[0].detach().numpy()
        plt.imsave(os.path.join(args.file_path,"before_mask_pred.png"), left_pred_bit_mask, cmap='gray')
        plt.imsave(os.path.join(args.file_path, "after_mask_pred.png"), right_pred_bit_mask, cmap='gray')
        print('Predictions saved to {}'.format(args.file_path))