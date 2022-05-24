import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def calc_ch(min_ch, max_ch, step):
    curr_ch = min_ch * (2**step)
    return max_ch if curr_ch > max_ch else curr_ch

class ConvRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ConvRelu, self).__init__(
            nn.Conv2d(in_ch, out_ch, (3,3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

class ConvSigmoid(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ConvSigmoid, self).__init__(
            nn.Conv2d(in_ch, out_ch, (1,1)),
            nn.Sigmoid()
        )

class DropConvRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(DropConvRelu, self).__init__(
            nn.Dropout(),
            nn.Conv2d(in_ch, out_ch, (3,3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

class BlockTemplate(nn.Module):
    def __init__(self):
        super(BlockTemplate, self).__init__()
        self.layer_list = nn.ModuleList()
    
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class Upsmaple(BlockTemplate):
    def __init__(self, min_ch, max_ch, step):
        super(Upsmaple, self).__init__()
        curr_ch = calc_ch(min_ch, max_ch, step)
        self.layer_list.append(nn.ConvTranspose2d(curr_ch, curr_ch, (3,3), 2, 1, 1))

class LastUp(BlockTemplate):
    def __init__(self, num_classes, min_ch, max_ch):
        super(LastUp, self).__init__()
        curr_ch = calc_ch(min_ch, max_ch, 1)

        self.layer_list.append(DropConvRelu(curr_ch * 2, curr_ch))
        self.layer_list.append(ConvRelu(curr_ch, min_ch))
        self.layer_list.append(ConvSigmoid(min_ch, num_classes))


class UpModule(BlockTemplate):
    def __init__(self, min_ch, max_ch, step):
        super(UpModule, self).__init__()
        curr_ch = calc_ch(min_ch, max_ch, step)
        next_ch = calc_ch(min_ch, max_ch, step - 1)
        # self.layer_list.append(ConvRelu(curr_ch * 2, next_ch))
        self.layer_list.append(DropConvRelu(curr_ch * 2, next_ch))
        self.layer_list.append(ConvRelu(next_ch, next_ch))

class FirstDown(BlockTemplate):
    def __init__(self, num_classes, min_ch, max_ch):
        super(FirstDown, self).__init__()
        next_ch = calc_ch(min_ch, max_ch, 1)
        
        self.layer_list.append(ConvRelu(num_classes, min_ch))
        self.layer_list.append(ConvRelu(min_ch, next_ch))
        # self.layer_list.append(ConvRelu(next_ch, next_ch))

class DownModule(BlockTemplate):
    def __init__(self, min_ch, max_ch, step):
        super(DownModule, self).__init__()
        curr_ch = calc_ch(min_ch, max_ch, step - 1)
        next_ch = calc_ch(min_ch, max_ch, step)
        self.layer_list.append(nn.MaxPool2d(2))
        self.layer_list.append(ConvRelu(curr_ch, next_ch))
        self.layer_list.append(ConvRelu(next_ch, next_ch))
        # self.layer_list.append(ConvRelu(next_ch, next_ch))

            
class BottleNeck(BlockTemplate):
    def __init__(self, min_ch, max_ch, step):
        super(BottleNeck, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.MaxPool2d(2))

        curr_ch = calc_ch(min_ch, max_ch, step)
        for i in range(3):
            self.layer_list.append(ConvRelu(curr_ch, curr_ch))



class UNet(nn.Module):

    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.layer_list = nn.ModuleList()
        self.num_down_blocks = num_down_blocks

        self.layer_list.append(FirstDown(3, min_channels, max_channels))

        for i in range(1, num_down_blocks - 1):
            self.layer_list.append(DownModule(min_channels, max_channels, i+1))

        self.layer_list.append(BottleNeck(min_channels, max_channels, num_down_blocks-1))
        
        for i in range(num_down_blocks - 1, 1, -1):
            self.layer_list.append(Upsmaple(min_channels, max_channels, i))
            self.layer_list.append(UpModule(min_channels, max_channels, i))
        
        self.layer_list.append(Upsmaple(min_channels, max_channels, 1))
        self.layer_list.append(LastUp(num_classes, min_channels, max_channels))
        
    def fix_size(self, x):
        orig_size = list(x.shape[2:])
        i, j = 0, 0 
        if x.shape[-1] % 2 != 0:
            i+=1
        if x.shape[-2] % 2 != 0:
            j+=1
        if i or j:
            orig_size[1] += i
            orig_size[0] += j
            x = F.upsample_bilinear(x, orig_size)
        # print(i, j, x.shape)
        return x

    def forward(self, inputs):
        trace = []
        orig_size = inputs.shape[2:]
        inputs = self.fix_size(inputs)

        layer = self.layer_list[0]
        x = layer(inputs)
        x = self.fix_size(x)
        trace.append(x)

        for i in range(1, self.num_down_blocks - 1):
            x = self.layer_list[i](x)
            x = self.fix_size(x)
            trace.append(x)

        x = self.layer_list[self.num_down_blocks-1](x)
        for i in range(0, self.num_down_blocks - 2):
            x = self.layer_list[self.num_down_blocks + 2*i](x)
            x = self.layer_list[self.num_down_blocks + 2*i + 1](torch.cat((trace[-i-1], x),1))

        x = self.layer_list[-2](x)
        if x.shape[2:] != orig_size:
            x = F.upsample_bilinear(x, orig_size)
        # print(x.shape, trace[0].shape, orig_size)
        logits = self.layer_list[-1](torch.cat((trace[0], x),1))
        if logits.shape[2:] != orig_size:
            logits = F.upsample_bilinear(logits, orig_size)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits
