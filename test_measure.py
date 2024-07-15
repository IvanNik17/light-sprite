import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import re

from model import Unet
from skimage.metrics import structural_similarity
import time

resize_amount = 128

transform = transforms.Compose([
    transforms.Resize((resize_amount, resize_amount)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # need to test this?
])



load_path = "light_checkpoints\...(Trained Model here)"

model = Unet()
model.to("cuda")


model.load_state_dict(torch.load(load_path)) 
model.eval()

is_dir_lighty = True

if is_dir_lighty:
    light_dir_path = "angles.txt"
else:
    light_dir_path = "positions.txt"

light_dirs  =  np.loadtxt(light_dir_path, delimiter=",", dtype= float)

print(light_dirs)


baseFolder = "test_inputs"

sub_folders = [name for name in os.listdir(baseFolder) if os.path.isdir(os.path.join(baseFolder, name))]


outputFolder = "results"

print(sub_folders)

ssim_scores = []

all_times = []

for curr_image in sub_folders:


    inputDir = f"test_inputs\{curr_image}"

    rgbDir = os.path.join(".", inputDir, f"{curr_image}.png")

    curr_output_main = os.path.join(".", outputFolder, curr_image)
    if not os.path.exists(curr_output_main):
        os.makedirs(curr_output_main)



    for path, subdirs, files in os.walk( os.path.join(".", inputDir, "target_images"), topdown=True):

        # print(files)

        test_image = Image.open(rgbDir).convert("RGB")
        test_image_base = np.asarray(test_image.copy())/255.0
        test_image_base = cv2.resize(test_image_base, (resize_amount, resize_amount))
        test_image = transform(test_image).unsqueeze(0).to("cuda") 

        counter = 0
        for file in files:


            curr_dir_img = cv2.imread(os.path.join(".", path, file))
            curr_dir_img = cv2.resize(curr_dir_img, (resize_amount, resize_amount))

            dir = light_dirs[counter]

            angle = dir[0:3] 
            angle_tensor = torch.FloatTensor(angle).unsqueeze(0).to("cuda")

            start_time = time.time()

            with torch.no_grad():
                output = model(test_image, angle_tensor)

            end_time = (time.time() - start_time)
            all_times.append(end_time)

            output_image = output.squeeze().cpu().permute(1, 2, 0).numpy() 

            output_image = (output_image-np.min(output_image))/(np.max(output_image)-np.min(output_image))

            gen_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            real_gray = cv2.cvtColor(curr_dir_img, cv2.COLOR_BGR2GRAY)/255



            (score, diff) = structural_similarity(gen_gray, real_gray, full=True, data_range = 1)
            print(f"Image {curr_image}, at angle {angle} Image similarity - {score}")


            ssim_scores.append(score)

            diff = (diff * 255).astype("uint8")



            filename = file.split(".")[0]
            cv2.imwrite(os.path.join(".", curr_output_main, f"{filename}_gen.jpg"), gen_gray*255)
            cv2.imwrite(os.path.join(".", curr_output_main, f"{filename}_real.jpg"), real_gray*255)

            cv2.imwrite(os.path.join(".", curr_output_main, f"{filename}_diff.jpg"), diff)

            counter+=1


            

np.savetxt(os.path.join(baseFolder, "ssim_results.txt"), ssim_scores, delimiter=',')

all_times_arr = np.array(all_times)

print(f"avg time per image {all_times_arr.mean()}")