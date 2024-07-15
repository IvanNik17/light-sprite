import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# from model_cbn import Unet

from model import Unet

from torchvision.utils import save_image

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.angle_vectors = []
        self.target_images = []
        self.input_images = []
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            
            if os.path.isdir(folder_path):
                input_image_path = os.path.join(folder_path, f"{folder}.png")
                angle_file_path = os.path.join(folder_path, "angles.txt")
                target_image_folder = os.path.join(folder_path, "target_images")
                
                
                if os.path.exists(input_image_path) and os.path.exists(angle_file_path) and os.path.exists(target_image_folder):
                    
                    with open(angle_file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) == 4:
                                yaw, pitch, roll, _ = map(float, parts)
                                self.angle_vectors.append(torch.FloatTensor([yaw, pitch, roll]))

                    # Load target images
                    
                    for i in range(len(self.angle_vectors)):
                        target_image_path = os.path.join(target_image_folder, f"illum_{i:03d}.jpg")
                        if os.path.exists(target_image_path):
                            target_image = Image.open(target_image_path).convert("RGB")
                            self.target_images.append(target_image)

                            input_image = Image.open(input_image_path).convert("RGB")
                            self.input_images.append(input_image)


    def __len__(self):
        return len( self.input_images)

    def __getitem__(self, idx):

        input_image  = self.input_images[idx]
        angle_vector  = self.angle_vectors[idx]
        target_image  = self.target_images[idx]

        # Apply transformations if specified
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, angle_vector, target_image

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to desired size
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define root directory containing image folders
root_dir = "inputs"

save_dir = "light_checkpoints"

# Create custom dataset
custom_dataset = CustomDataset(root_dir, transform=data_transform)

# Define DataLoader
batch_size = 4
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

model = Unet()
model.to("cuda")

criterion = nn.MSELoss().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print(len(data_loader))

model.train()

print_freq = 10
num_epochs = 50
all_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0 
    for i,(input_images, angle_vectors, target_images) in enumerate(data_loader):
        
        input_images = input_images.to("cuda")
        angle_vectors = angle_vectors.to("cuda")
        target_images = target_images.to("cuda")

        outputs = model(input_images, angle_vectors)
        
        loss = criterion(outputs, target_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_images.size(0)


        if i % print_freq == print_freq - 1:
                batch_loss = running_loss / (print_freq * input_images.size(0))
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], Loss: {batch_loss:.4f}')
                running_loss = 0.0
                all_losses.append(batch_loss)
        
        epoch_loss += loss.item() * input_images.size(0) 

    if epoch % 10 == 0:
        save_image(outputs[0], os.path.join(save_dir,"show_imgs", f'img_epoch_{epoch}.png'), value_range=(-1,1), normalize=True, nrow=batch_size/2 )

    epoch_loss /= len(custom_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    

    # Save the model after each epoch
    model_path = os.path.join(save_dir, f'unet_model_epoch_{epoch+1}.pth') 
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as {model_path}')
    
np.savetxt(os.path.join(save_dir, "losses.txt"), all_losses, delimiter=',', fmt='%f')
