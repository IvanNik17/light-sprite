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
import time

from model import Unet

def interpolate_rows(arr, num_interpolations=1):
    interpolated_array = []
    
    for i in range(arr.shape[0] - 1):
        start_point = arr[i]
        end_point = arr[i + 1]
        
        for j in range(num_interpolations + 1):
            interp_point = start_point + (end_point - start_point) * (j / (num_interpolations + 1))
            interpolated_array.append(interp_point)
    
    interpolated_array = np.array(interpolated_array)
    
    return interpolated_array


resize_amount = 128
# Define any necessary transformations
transform = transforms.Compose([
    transforms.Resize((resize_amount, resize_amount)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # need to test this?
])


load_path = "light_checkpoints\...(Trained Model here)"

img_name = "8_6980.png"

image_path = f"test_inputs\{img_name}"


is_dir_lighty = True

if is_dir_lighty:
    light_dir_path = "angles.txt"
else:
    light_dir_path = "positions.txt"

light_dirs  =  np.loadtxt(light_dir_path, delimiter=",", dtype= float)


num_interpolations = 3 

light_dirs = interpolate_rows(light_dirs, num_interpolations)

print(light_dirs)

model = Unet()
model.to("cuda")


model.load_state_dict(torch.load(load_path)) 
model.eval()



test_image = Image.open(image_path).convert("RGB")
test_image_base = np.asarray(test_image.copy())/255.0
test_image_base = cv2.resize(test_image_base, (resize_amount, resize_amount))
test_image = transform(test_image).unsqueeze(0).to("cuda") 



for dir in light_dirs:

    start_time = time.time()

    print(dir)
    angle = dir[0:3] 
    angle_tensor = torch.FloatTensor(angle).unsqueeze(0).to("cuda")

    print(angle_tensor)


    with torch.no_grad():
        output = model(test_image, angle_tensor)

    output_image = output.squeeze().cpu().permute(1, 2, 0).numpy() 

    output_image = (output_image-np.min(output_image))/(np.max(output_image)-np.min(output_image))

    print("--- %s seconds ---" % (time.time() - start_time))
    


    cv2.imshow(f"Output", output_image)



    cv2.imshow(f"Lit", np.flip(test_image_base , axis=-1)  * output_image + np.ones_like(output_image)*0.1)

    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break

