"""
这是用于阴影线性恢复的UNet模型
输入：4通道（3通道RGB图片 + 1通道阴影mask）
输出：3通道（去除阴影后的RGB图片）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # 定义一个双卷积层
    def __init__(self, in_channels, out_channels):
        # 初始化函数，传入输入通道数和输出通道数
        super(DoubleConv, self).__init__()
        # 调用父类的初始化函数
        self.double_conv = nn.Sequential(
            # 定义一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3，padding为1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 定义一个批归一化层，输入通道数为out_channels
            nn.BatchNorm2d(out_channels),
            # 定义一个ReLU激活函数，inplace为False
            nn.ReLU(inplace=False),
            # 定义另一个卷积层，输入通道数为out_channels，输出通道数为out_channels，卷积核大小为3，padding为1
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # 定义另一个批归一化层，输入通道数为out_channels
            nn.BatchNorm2d(out_channels),
            # 定义另一个ReLU激活函数，inplace为False
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        # 定义前向传播函数，传入输入x
        return self.double_conv(x)


class Down(nn.Module):
    # 定义一个下采样类，继承自nn.Module
    def __init__(self, in_channels, out_channels):
        # 初始化函数，传入输入通道数和输出通道数
        super(Down, self).__init__()
        # 调用父类的初始化函数
        self.maxpool_conv = nn.Sequential(
            # 定义一个序列容器，包含两个操作：最大池化和双卷积
            nn.MaxPool2d(2),
            # 最大池化操作，池化核大小为2
            DoubleConv(in_channels, out_channels)
            # 双卷积操作，输入通道数为in_channels，输出通道数为out_channels
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Up sampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        # 因为x1比x2多进行一次下采样，如果拼接需要将x1先进行上采样
        x1 = self.up(x1)

        # Ensure x1 and x2 have the same spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)  
        return self.conv(x)


class ShadowRemovalUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(ShadowRemovalUnet, self).__init__()
        self.in_channels = in_channels  # 4通道：RGB + 阴影mask
        self.out_channels = out_channels  # 3通道：无阴影RGB

        # 编码器部分
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器部分
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 线性恢复层
        self.linear_restore = nn.Sequential(
            nn.Conv2d(out_channels + 1, 32, kernel_size=3, padding=1),  # +1 为阴影mask
            nn.ReLU(inplace=False),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 分离输入图像和阴影mask
        input_image = x[:, :3, :, :]  # RGB图像
        shadow_mask = x[:, 3:4, :, :]  # 阴影mask
        
        # 编码器前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器前向传播
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 生成初步输出
        output = self.outc(x)
        
        # 线性恢复: 使用输出和阴影mask进一步优化
        # 将mask与输出连接起来进行最终调整
        combined = torch.cat([output, shadow_mask], dim=1)
        final_output = input_image + self.linear_restore(combined)
        
        return final_output

    def use_checkpointing(self):
        # 使用checkpointing技术，减少内存占用
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.linear_restore = torch.utils.checkpoint(self.linear_restore)


if __name__ == '__main__':
    # 测试模型
    model = ShadowRemovalUnet(in_channels=4, out_channels=3)
    input_tensor = torch.rand((2, 4, 256, 256))  # 批量大小2，4通道输入，256x256分辨率
    output = model(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 确保输出大小正确
    assert output.shape == (2, 3, 256, 256), f"输出形状错误: {output.shape}, 预期: (2, 3, 256, 256)"