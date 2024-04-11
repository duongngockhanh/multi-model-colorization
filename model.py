import torch.nn as nn
import torch
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, previous_output, skip_output):
        x = self.conv_trans(previous_output)
        output = torch.cat([x, skip_output], dim=1)
        output = self.conv(output)
        return output

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten prediction and target tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = torch.sum(input_flat * target_flat)
        dice_coeff = (2. * intersection + self.smooth) / (torch.sum(input_flat) + torch.sum(target_flat) + self.smooth)

        return 1 - dice_coeff

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_clas=365, out_reg=2, out_soft=313, out_seg=183):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, 64)

        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)

        # H_clas
        self.clas_avgpool   = nn.AdaptiveAvgPool2d(1)
        self.clas_flatten   = nn.Flatten()
        self.clas_fc        = nn.Linear(1024, out_clas)
        self.clas_softmax   = nn.Softmax(dim=1)

        # H_reg
        self.reg_conv = nn.Conv2d(64, out_reg, kernel_size=1)
        self.reg_tanh = nn.Tanh()

        # H_soft
        self.soft_conv = nn.Conv2d(64, out_soft, kernel_size=1)
        self.soft_softmax = nn.Softmax(dim=1)

        # H_seg
        self.seg_conv = nn.Conv2d(64, out_seg, kernel_size=1)
        self.seg_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)

        # H_clas
        x_clas = self.clas_avgpool(x5)
        x_clas = self.clas_flatten(x_clas)
        x_clas = self.clas_fc(x_clas)
        x_clas = self.clas_softmax(x_clas)

        # H_reg
        x_reg = self.reg_conv(x)
        x_reg = self.reg_tanh(x_reg)

        # H_soft
        x_soft = self.soft_conv(x)
        x_soft = self.soft_softmax(x_soft)

        # H_seg
        x_seg = self.seg_conv(x)
        x_seg = self.seg_softmax(x_seg)

        return x_clas, x_reg, x_soft, x_seg
    

if __name__ == "__main__":
    model = UNet()
    # summary(model, (1, 256, 256))
    img = torch.rand(1, 1, 256, 256)
    x_clas, x_reg, x_soft, x_seg = model(img)
    print(x_clas.shape)
    print(x_reg.shape)
    print(x_soft.shape)
    print(x_seg.shape)