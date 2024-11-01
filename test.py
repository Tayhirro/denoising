import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse


from src.models import unet  # Assuming UNet is defined in this path

# Define the test transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_model(model_path, device):
    # Load the model
    model = unet.UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def denoise_image(model, device, img_path, output_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
    
    # Convert output to image format and save
    output_img = output.squeeze().cpu().numpy()  # Remove batch dimension
    if output_img.shape[0] == 1:
        output_img = np.tile(output_img, (3, 1, 1))  # Repeat channel if only one channel exists

    output_img = np.transpose(output_img, (1, 2, 0)) * 255  # Change from (C, H, W) to (H, W, C)
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save(output_path)

def main (args):

    #####################################
    test_model_path =args.test_model_path 
    test_image_dir = 'C:/Users/Tayhirro/Desktop/Test/noise'
    output_dir = 'C:/Users/Tayhirro/Desktop/Test/outputs'
    #####################################


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(test_model_path, device)

    # Process all test images
    for img_name in os.listdir(test_image_dir):
        img_path = os.path.join(test_image_dir, img_name)
        output_path = os.path.join(output_dir, f'denoised_{img_name}')
        denoise_image(model, device, img_path, output_path)
        print(f'Saved denoised image to {output_path}')

    print("Testing complete. Denoised images saved in:", output_dir)



if __name__ == "__main__":
    # Define paths

    parser=argparse.ArgumentParser(description='PIC denoise test')    

    parser.add_argument('--test_model_path',type=str)
    args=parser.parse_args()
    main(args)