import torch,time,os
from torch import nn, optim
from torchvision import transforms
import numpy as np
import argparse

#from options import Options

import datasets
from src.models import unet
from PIL import Image
from datasets.Denoising_dataset import DenoisingDataset


train_root = 'C:/Users/Tayhirro/Desktop/Train'
test_root = 'C:/Users/Tayhirro/Desktop/Test'

def train(model, loader, criterion, optimizer, epochs, device, save_path):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        imgsum = 0
        for noisy_imgs, clean_imgs in loader:
            imgsum += 1
            print(f"now {imgsum} is loading")
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            model_save_path = f'{save_path}/unet_denoising_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')



def main(args):
    args.seed=1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 预处理并保存数据集为 .npy 格式
    noisy_dir = os.path.join(train_root, 'noise')
    clean_dir = os.path.join(train_root, 'orign')

    noisy_data = []
    clean_data = []
    for img_name in sorted(os.listdir(noisy_dir)):
        img_path = os.path.join(noisy_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        noisy_data.append(img.numpy())

    np.save('noisy_data.npy', np.array(noisy_data))

    for img_name in sorted(os.listdir(clean_dir)):
        img_path = os.path.join(clean_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        clean_data.append(img.numpy())
    
    np.save('clean_data.npy', np.array(clean_data))

    dataset_func = DenoisingDataset  # 确保指向 DenoisingDataset 类


    train_dataset=dataset_func('noisy_data.npy','clean_data.npy')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)



    # 定义模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(in_channels=3, out_channels=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 设置模型保存路径
    save_path = 'C:/Users/Tayhirro/Desktop/models'
    train(model, train_loader, criterion, optimizer, epochs=500, device=device, save_path=save_path)



# 主程序入口
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    #parser=Options().init(argparse.ArgumentParser(description='PIC denoise'))
    parser=argparse.ArgumentParser(description='PIC denoise')
    args = parser.parse_args()
    #暂时不处理args

    main(args)
    