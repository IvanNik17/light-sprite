import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import glob



def light_pos_cube(cube_size, num_points, depth_padding = 4):
    
    x = np.linspace(0, cube_size, num_points)
    y = np.linspace(0, cube_size, num_points)
    z = np.linspace(0, cube_size, num_points)
    xv, yv, zv = np.meshgrid(x, y, z)
    
    zv +=depth_padding
    points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    return points

def compute_normals(depth):
    dzdx = np.gradient(depth, axis=1)
    dzdy = np.gradient(depth, axis=0)

    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= norm
    normal[:, :, 1] /= norm
    normal[:, :, 2] /= norm

 
    return normal


def phong_reflection(light_direction, view_dir, surface_normal, shininess):
    light_direction = light_direction / np.linalg.norm(light_direction)
    view_dir = view_dir / np.linalg.norm(view_dir)
    surface_normal = surface_normal / np.linalg.norm(surface_normal)

    reflection_direction = -light_direction + 2 * np.dot(light_direction, surface_normal) * surface_normal
    diffuse = np.maximum(np.dot(surface_normal, light_direction), 0)

    specular = np.maximum(np.dot(reflection_direction, view_dir.T), 0) ** shininess


    return diffuse, specular


def apply_point_light(depth, light_pos, view_pos = np.array([0, 0, 40]),shininess= 10, ambient=0.1, diffuse_percept=1,specular_percent = 1, K_c=1.0, K_l=0.045, K_q=0.0075):
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    normals = compute_normals(depth)
    positions = np.dstack((x, y, depth))
    light_vectors = light_pos - positions

    light_distances = np.linalg.norm(light_vectors, axis=2, keepdims=True)
    light_directions = light_vectors / light_distances
    view_direction = view_pos - positions
    view_distances = np.linalg.norm(view_direction, axis=2, keepdims=True)
    view_direction /= view_distances

    attenuation = 1.0 / (K_c + K_l * light_distances + K_q * light_distances ** 2)

    diffuse_intensities = np.zeros_like(depth)
    specular_intensities = np.zeros_like(depth)

    for i in range(height):
        for j in range(width):
            surface_normal = normals[i, j]
            light_direction = light_directions[i, j]
            view_dir = view_direction[i, j]
            
            diffuse, specular = phong_reflection(light_direction, view_dir, surface_normal, shininess)
            diffuse_intensities[i, j] = diffuse
            specular_intensities[i, j] = specular

    total_intensity = ambient + diffuse_intensities* attenuation.squeeze() + specular_intensities

    total_intensity = np.clip(total_intensity, 0, 1)
    return total_intensity


view_position = np.array([0, 0, 100])  

shininess = 30

light_pos = light_pos_cube(16, 4)


shininess_steps = [shininess]
# shininess_steps = np.arange(10,60,10)

inputDir = os.path.join("inputs/depth")

inputDir_rgb = os.path.join("inputs/rgb")

outputDir = os.path.join("outputs")


for path, subdirs, files in os.walk( os.path.join(".", inputDir), topdown=True):
				
        
        if len(files) == 0:
            continue

        for file in files:
            
            sep_name = file.split(".")
            # print(os.path.join(path, file))
            # print(sep_name)
            output_curr = os.path.join(outputDir, sep_name[0])
            output_curr_imgs = os.path.join(output_curr,"target_images")
            
            if not os.path.isdir(output_curr):
                os.mkdir(output_curr)
            if not os.path.isdir(output_curr_imgs):
                os.mkdir(output_curr_imgs)

            
            depth_image = cv2.imread(os.path.join(path, file))
            depth_image = np.float32(depth_image[:,:,0])
            depth_image = (depth_image / depth_image.max()) * depth_image.shape[0]

            angles_shine = []
            
            cntr = 0
            for i in range(0, len(light_pos)):
                # Initial light position
                initial_light_position = np.array([light_pos[i,0], light_pos[i,1], light_pos[i,2]])
                # Apply Phong lighting model
                for curr_shine in shininess_steps:
                    shaded_image = apply_point_light(depth_image, initial_light_position, view_position, curr_shine)
                    shaded_image = (shaded_image * 255).astype(np.uint8)
                    curr_illum_name = f"illum_{str(cntr).zfill(3)}.jpg"

                    angles_shine.append([light_pos[i,0]/float(depth_image.shape[0]), light_pos[i,1]/float(depth_image.shape[0]), light_pos[i,2]/(float(depth_image.shape[0])+4), curr_shine])

                    cv2.imwrite(os.path.join(output_curr_imgs, curr_illum_name),shaded_image)

                    cntr +=1

            np.savetxt(os.path.join(output_curr, "angles.txt"), angles_shine, delimiter=',', fmt='%f')

            rgb_img = cv2.imread(os.path.join(inputDir_rgb, file))
            cv2.imwrite(os.path.join(output_curr, file),rgb_img)

            print(f"Image {file} finished")


