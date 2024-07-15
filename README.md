# Light Sprite
UNet Pixel Art Dynamic Light Generation

This repository contains the code for the paper "Light the Sprite: Pixel Art Dynamic Light Generation" for training a UNet-based deep learning model with Conditional instance normalization layers (CIN) for producing combined dynamic directional and point light map with normal map surface information.

| Point Light | Directional Light |
|-------------|-------------------|
| ![Visual](example_imgs/naruto_gif.gif)            |   ![Visual](example_imgs/mummy_dir_gif.gif)                |
|  ![Visual](example_imgs/kirby_gif.gif)           |    ![Visual](example_imgs/palm_dir_gif.gif)               |



# Requirements

- PyTorch >2.0, torchvision
- OpenCV
- Pillow
- Numpy

