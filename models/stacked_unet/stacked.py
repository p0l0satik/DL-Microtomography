#import torch
#from torch import nn
#from torch.nn import functional as F
#from torchvision import models
# from msilib.schema import Class
import numpy as np
import torch.nn.functional as F

from UNet_parts import *



class UNet(nn.Module):
    """
    TODO: 8 points
    A standard UNet network (with padding in covs).

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 in_channels=31,
                 out_channels=2,
                 min_channels= 64, #64, #32,
                 max_channels= 512, # 324,
                 num_down_blocks=5): #4):
        super(UNet, self).__init__()
        self.out_channels = out_channels
        # TODO
        #in_channels = 30 #razoral
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        ########################################
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_down_blocks =  num_down_blocks
        
        self.inc = DoubleConv(in_channels, min_channels)
        self.inc = self.inc.to(device)

        self.outc = OutConv(min_channels, in_channels)
        self.outc = self.outc.to(device)
        
        # DOWN list
        down_list = nn.ModuleList()
        down_chan = []
        down_list.append(self.inc)
        down_chan.append( self.inc.out_channels )

        for block in range(num_down_blocks):

            in_chan = int(min_channels*np.power(2, block))
            if in_chan > max_channels:
              in_chan = max_channels

            out_chan = int(min_channels*np.power(2, block+1))
            if out_chan > max_channels:
              out_chan = max_channels

            down_chan.append(out_chan)

            block_cur = Down(in_chan, out_chan )
            block_cur = block_cur.to(device)
            
            down_list.append(block_cur)
        self.down = down_list

        down_chan.reverse()

        up_list = nn.ModuleList()

        for block in range(num_down_blocks-1):

          chan_in1 = down_chan[block]
          chan_in2 = down_chan[block+1]
          chan_out = down_chan[block+1] 

          block_cur = Up(chan_in1, chan_in2, chan_out)
          block_cur = block_cur.to(device)
        
          up_list.append(block_cur)         
          
        chan_in1 = down_chan[-2]
        chan_in2 = down_chan[-1]
        chan_out = down_chan[-1]

        block_cur = Up(chan_in1, chan_in2, chan_out)
        block_cur = block_cur.to(device)
        
        up_list.append(block_cur)

        self.up = up_list
    def forward(self, inputs):

        x = inputs.clone()
        x_down_ = []
        for i, down_block in enumerate(self.down):

          x = down_block(x)
          #print("x{}_down:".format(i+1), x.shape)

          x_down_.append( x ) #x_inc, x_down1, x_down2, ... x_bottleneck 
          
        #print("len x_down_", len(x_down_))

        up_block = self.up[0]
                        #bottleneck_x   #last_down_x 
        x_up = up_block( x_down_[-1], x_down_[-2] )
        x_down_.pop(); x_down_.pop();
        #print("\nx{}_up:".format(1),  x_up.shape)
        #print("\n up")
        for j, up_block in enumerate(self.up[1:]):

          x_up = up_block(x_up, x_down_.pop() )

        logits = self.outc(x_up)
        #print("logits:", logits.shape)

        assert logits.shape == (inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits

class Stacked(nn.Module):
    def __init__(self, 
                 in_channels=31,
                 out_channels=2,
                 n_stacks = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_stacks = n_stacks

        self.blocks = nn.ModuleList()
        self.intermediate_supervision = nn.ModuleList()
        self.remap = nn.ModuleList()

        for _ in range(n_stacks):
            self.blocks.append(UNet(out_channels=31, num_down_blocks=4))
            self.intermediate_supervision.append(
                # nn.Sequential(
                #     OutConv(in_channels, out_channels),
                #     nn.Conv2d(2, 2, kernel_size = 1, stride = 1)
                # )
                OutConv(in_channels, out_channels)
            )
            self.remap.append(nn.Conv2d(out_channels, in_channels, kernel_size = 1, stride = 1))
        
    def forward(self, x):
        output = [] #all intermediate supervisions
        for i in range(self.n_stacks):
            out1 = self.blocks[i](x) #31->31 ch
            out2 = self.intermediate_supervision[i](out1) #31->2 for supervision
            out2 = F.sigmoid(out2)
            output.append(out2)
            out2_remap = self.remap[i](out2) #2->31 for continuation
            x = out1 + out2_remap + x
        

        return output


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        
        self.CBR = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.1),
        )
        
    def forward(self, x):
        return self.CBR(x)

class BaselineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(ConvBNReLU(31, 64),
                                 ConvBNReLU(64, 128),
                                 ConvBNReLU(128, 64),
                                 ConvBNReLU(64, 32))
    def forward(self, x):
        return self.net(x)

class StackedBaseline(nn.Module):

    def __init__(self, 
                in_channels=31,
                out_channels=2,
                n_stacks = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_stacks = n_stacks

        self.blocks = nn.ModuleList()
        self.intermediate_supervision = nn.ModuleList()
        self.remap = nn.ModuleList()

        for _ in range(n_stacks):
            self.blocks.append(BaselineNetwork())
            self.intermediate_supervision.append(
                # nn.Sequential(
                #     OutConv(in_channels, out_channels),
                #     nn.Conv2d(2, 2, kernel_size = 1, stride = 1)
                # )
                ConvBNReLU(in_channels, out_channels)
            )
            self.remap.append(nn.Conv2d(out_channels, in_channels, kernel_size = 1, stride = 1))
            
    def forward(self, x):
        output = [] #all intermediate supervisions
        for i in range(self.n_stacks):
            out1 = self.blocks[i](x) #31->31 ch
            out2 = self.intermediate_supervision[i](out1) #31->2 for supervision
            output.append(out2)
            out2_remap = self.remap[i](out2) #2->31 for continuation
            x = out1 + out2_remap + x
        
        return output

