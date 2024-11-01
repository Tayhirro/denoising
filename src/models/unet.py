import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):  # 设置输入和输出通道数为 3
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            return block

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        up4 = self.upconv4(b)
        merge4 = torch.cat([up4, e4], dim=1)
        d4 = self.decoder4(merge4)

        up3 = self.upconv3(d4)
        merge3 = torch.cat([up3, e3], dim=1)
        d3 = self.decoder3(merge3)

        up2 = self.upconv2(d3)
        merge2 = torch.cat([up2, e2], dim=1)
        d2 = self.decoder2(merge2)

        up1 = self.upconv1(d2)
        merge1 = torch.cat([up1, e1], dim=1)
        d1 = self.decoder1(merge1)

        out = self.conv_last(d1)
        return out