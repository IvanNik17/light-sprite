import numpy as np
import matplotlib.pyplot as plt

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

def get_direction(mouse_x, mouse_y, image_width, image_height, z=1):

    center_x = image_width / 2
    center_y = image_height / 2
    center_z = 0 


    mouse_pos = np.array([mouse_x, mouse_y, z])


    center_pos = np.array([center_x, center_y, center_z])

    direction = center_pos - mouse_pos

    norm = np.linalg.norm(direction)
    if norm != 0:
        direction = direction / norm

    return direction



resize_amount = 128

transform = transforms.Compose([
    transforms.Resize((resize_amount, resize_amount)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # need to test this?
])


is_dir = False


load_path = "light_checkpoints\...(Trained Model here)"


img_name = "5_3270.png"

image_path = f"test_inputs\{img_name}"


model = Unet()
model.to("cuda")


model.load_state_dict(torch.load(load_path)) 
model.eval()



test_image = Image.open(image_path).convert("RGB")
test_image_base = np.asarray(test_image.copy())/255.0
test_image_base = cv2.resize(test_image_base, (resize_amount, resize_amount))
test_image = transform(test_image).unsqueeze(0).to("cuda") 





def update_plot(event):
    if event.inaxes:

        start_time = time.time()
        light_position = np.array([event.xdata/128, event.ydata/128, 1]) 

        if is_dir == True:
            light_position = get_direction(event.xdata,event.ydata, 128, 128)
            light_position = -light_position
 
        angle_tensor = torch.FloatTensor(light_position).unsqueeze(0).to("cuda")
        with torch.no_grad():
            output = model(test_image, angle_tensor)

        output_image = output.squeeze().cpu().permute(1, 2, 0).numpy() 

        output_image = (output_image-np.min(output_image))/(np.max(output_image)-np.min(output_image))

        print("--- %s seconds ---" % (time.time() - start_time))



        if is_dir:
            lit_up_rgb_image = test_image_base * output_image 
        else:
            lit_up_rgb_image = test_image_base * output_image + np.ones_like(test_image_base)*0.1




        ax_image.set_array(lit_up_rgb_image)
        fig.canvas.draw_idle()



initial_light_position = np.array([resize_amount/2, resize_amount/2, 1])


shaded_image = np.zeros_like(test_image_base)


fig, ax = plt.subplots()
ax_image = ax.imshow(shaded_image, cmap='gray')
ax.set_title('Visualize by moving the mouse')


fig.canvas.mpl_connect('motion_notify_event', update_plot)


plt.show()