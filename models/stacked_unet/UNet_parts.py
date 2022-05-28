import torch
from torch import nn
from torch.nn import functional as F
#from torchvision import models

###################################################
class DoubleConv(nn.Module):
    """Conv2d --> [batch_norm] --> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print("\n\t double conv x\n", x.shape)
        #print("\n\t x_in=", x.shape)

        #x1 = nn.Conv2d(x, self.in_channels, self.out_channels, kernel_size=3, padding=1)
        #print("1 - nn.Conv2d done", x1.shape)

        #x2 = nn.BatchNorm2d(x1, self.out_channels)
        #print("BatchNorm2d done", x2.shape)
        #x2 = nn.ReLU(x2, inplace=True)
        #print("ReLU done", x2.shape)

        #x3 = nn.Conv2d(x2, self.out_channels, self.out_channels, kernel_size=3, padding=1),
        #print("2 - nn.Conv2d done", x3.shape)

        return self.double_conv(x)
###################################################
class Down(nn.Module):
    """downsampling block with Maxpool and DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # nn.MaxPool2d(3, stride=2) #nn.MaxPool2d((3, 2), stride=(2, 1))
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #print("\t in Down_class")
        #x_max = nn.MaxPool2d(2)
        #print("x_max", x_max.shape)
        #x_out = DoubleConv(x_max)
        #print("x_out", x_out.shape)    
        return self.maxpool_conv(x)
###################################################
class Up(nn.Module):
    """upsampling block: upscaling --> concat --> [dropout] --> DoubleConv"""
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()

        self.in_channels1 = in_channels1

        self.in_channels2 = in_channels2

        self.out_channels = out_channels

        self.up = nn.ConvTranspose2d(in_channels1, in_channels1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.dropout = nn.Dropout(p=0.5)

        self.conv = DoubleConv(in_channels1+in_channels2, out_channels)
    
    def forward(self, x1, x2):
        # up
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] 
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concat down_up       
        x = torch.cat([x2, x1], dim=1)

        # dropout
        x_drop = self.dropout(x)

        # doubleConv
        return self.conv(x_drop)
###################################################
class OutConv(nn.Module):
    "outConv"
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #self.sigmoid = nn.Sigmoid

    def forward(self, x):
        return self.conv(x)
###################################################