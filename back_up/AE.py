
#Full assembly of the parts to form the complete network#
#reference:https://github.com/milesial/Pytorch-UNet/blob/master/unet#
#changes number of layyers to 3 instead of 4

import torch.nn.functional as F

from AE_parts import *
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 )
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64 , bilinear)
        self.outc = OutConv(64, n_classes)

    def sparse_loss(self):
        loss = 0
        for class_obj in self.modules():
            if isinstance(class_obj, Up):
                for module_up in class_obj.modules():
                    if isinstance(module_up, nn.Conv2d):
                        loss += torch.mean((module_up.weight.data.clone()) ** 2)
                #for j in range(len(model_children[i])):
            #values = F.relu((model_children[i](values)))
                #loss += torch.mean((values)**2)
                #loss=0
        return loss

    def weight_reset(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.outc(x)
        return logits,x4