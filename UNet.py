class UNet(nn.Module):

  def __init__(self, 
                 num_classes=2,
                 inp_channels=31,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.num_down_blocks = num_down_blocks

        def conv(num, direction):
          if direction=='down':
            in_1 = inp_channels if num==0 else min_channels*(2**num)
            out_1 = in_1*2 if (num!=num_down_blocks and num!=0) else (min_channels if num==0 else in_1)
            out_2 = out_1 if num!=0 else out_1*2
          else:
            in_1 = min_channels*(2**(num+2))
            out_1 = int(in_1/4) if num!=0 else int(in_1/2)
            out_2 = out_1 if num!=0 else int(out_1/2)

          conv = nn.Sequential(
              nn.Conv2d(in_channels = in_1, out_channels = out_1, kernel_size=3, padding=1, padding_mode='replicate'),
              nn.BatchNorm2d(num_features = out_1),
              nn.ReLU(),
              nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size=3, padding=1, padding_mode='replicate'),
              nn.BatchNorm2d(num_features = out_2),
              nn.ReLU()
          )
          return conv

        def conv_trans(num):
          num_channels = min_channels * (2**(num+1))
          conv_trans = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode='zeros')
          return conv_trans


        self.downs = nn.ModuleList([conv(i, 'down') for i in range(num_down_blocks+1)])
        self.ups = nn.ModuleList([conv(i, 'up') for i in range(num_down_blocks)])

        self.trans = nn.ModuleList([conv_trans(i) for i in range(num_down_blocks)])

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout2d(p=0.5)

        self.final = nn.Conv2d(in_channels = min_channels, out_channels = num_classes, kernel_size=1)

  def forward(self, inputs):

        initial_div, _initial_div = inputs.shape[2], inputs.shape[3]
        div = 2**self.num_down_blocks
        if initial_div % div != 0:
          new_size = round(initial_div/div) * div
          inputs = F.interpolate(inputs, size=new_size)

        down_outs_list = [0]*(self.num_down_blocks+2)
        down_outs_list[0] = inputs
        x = inputs

        for i in range(len(self.downs)-1):
          down_outs_list[i+1] = self.downs[i](x)
          x = self.maxpool(down_outs_list[i+1])

        prev = self.downs[-1](self.maxpool(down_outs_list[self.num_down_blocks]))
        down_outs_list[self.num_down_blocks+1] = self.trans[-1](prev)
        x = down_outs_list[-1]
        for i in range(self.num_down_blocks-1, 0, -1):
          med = self.ups[i](self.dropout(torch.cat((x, down_outs_list[i+1]), 1)))
          x = self.trans[i-1](med)

        logits = self.final(self.ups[0](self.dropout(torch.cat((x, down_outs_list[1]), 1))))

        logits = torch.sigmoid(F.interpolate(logits, size=(initial_div, _initial_div)))

        p = torch.permute(logits, (1, 0, 2, 3))

        au = p[0]*15
        al = p[1]*150

        logits = torch.stack((au, al), dim=1)

        del down_outs_list 

        assert logits.shape == (inputs.shape[0], self.num_classes, initial_div, _initial_div), 'Wrong shape of the logits'
        return logits
