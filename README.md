# [Remote sensing image change detection](https://sites.google.com/vt.edu/image-change-detection)

Harish Babu Manogaran, Mridul Khurana and Shri Sarvesh Venkatachala Moorthy

Fall 2022 ECE 4554/5554 Computer Vision: Course Project - Virginia Tech

## Datasets

### SECOND Dataset

[Download](https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view?usp=sharing)
```
second_dataset
└───SECOND_train_set
│   │   data_split.pkl
│   │   image_full_paths.txt
│   │
│   └───im1
│   |   │   <index>.png (original image before)
│   |   │   ...
|   |
│   └───im2
│   |   │   <index>.png (original image after)
│   |   │   ...
|   |
│   └───label1
│   |   │   <index>.png (original image before label)
│   |   │   ...
|   |   |
│   └───label2
│   |   │   <index>.png (original image after label)
│   |   │   ...
│   
└───test
│   │   data_split.pkl
│   │   image_full_paths.txt
│   │
│   └───im1
│   |   │   <index>.png (original image before)
│   |   │   ...
|   |
│   └───im2
│   |   │   <index>.png (original image after)
│   |   │   ...
|   |
│   └───label1
│   |   │   <index>.png (original image before label)
│   |   │   ...
|   |   |
│   └───label2
│   |   │   <index>.png (original image after label)
│   |   │   ...
```

Note: You can get the `data_split.pkl` and `image_full_paths.txt` in the `second_dataset` folder uploaded here.


## Example Usage

Disclaimer - Don't forget to update the `path_to_dataset` in the relevant config files.

### Training:

`python main.py --method segmentation --gpus 1 --config_file configs/second_dataset.yaml --experiment_name segmentation_resnet18_3x_coam_layers_256_50_epochs --max_epochs 50 --decoder_attention scse `

The codebase is heavily tied in with [Pytorch Lightning](https://www.pytorchlightning.ai/) and [Weights and Biases](https://wandb.ai/r). You may find the following flags helpful:

- `--no_logging` (disables logging to weights and biases)
- `--quick_prototype` (runs 1 epoch of train, val and test cycle with 2 batches)
- `--resume_from_checkpoint <path>`
- `--load_weights_from <path>` (initialises the model with these weights)
- `--wandb_id <id>` (for weights and biases)
- `--experiment_name <name>` (for weights and biases)

### Model Checkpoint
Please use the following to download the datasets presented in this work. The checksums can be found at `checkpoints/segmentation_model.ckpt`


### Demo:
We added some sample images to  `sample_images` folder which can be viewed and can run the demo for prediction using the code below:

`python demo.py --load_weights_from checkpoints/segmentation_model.ckpt --config_file configs/second_dataset.yaml --decoder_attention scse --file_path sample_images/2`

For simplicity you can run the `demo.ipynb` notebook directly on colab

## References 
 We have builded upon the code of the original paper [The Change You Want to See](https://arxiv.org/pdf/2209.14341.pdf) for which the code can be found [here](https://github.com/ragavsachdeva/The-Change-You-Want-to-See)