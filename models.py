import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1=nn.Conv2d(in_planes, planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True) # bias=trucnated random([24], [24], ..., [6])
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes, planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential()
        if stride!=(1,1):
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(3,3), stride=(1,1), bias=True),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=F.relu(out)
        out+=self.shortcut(x)
        return out

class TH(nn.Module):
    def __init__(self, block, num_blocks):
        super(TH, self).__init__()
        self.in_planes=4
        self.planes=24
        self.out_planes=6

        self.conv1=nn.Conv2d(self.in_planes, self.planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        self.bn1=nn.BatchNorm2d(self.planes)
        self.conv2=nn.Conv2d(self.planes, self.out_planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        self.bn2=nn.BatchNorm2d(self.out_planes)
        self.layer1=self._make_layer(block, 24, num_blocks[0], stride=(1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides=[stride]+[(1,1)]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(planes, planes, stride))
            self.in_planes=planes
        return nn.Sequential(*layers)
        
    def forward(self, x, y):
        input=torch.cat([x, y], dim=1)
        out=F.relu(self.bn1(self.conv1(input)))
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=self.layer1(out)
        #print('out=', out.shape)
        out=torch.tanh(self.bn2(self.conv2(out)))
        amp=torch.add(out[:,:3,:,:]*np.sqrt(0.5), np.sqrt(0.5))
        phs=torch.add(out[:,3:,:,:]*0.5, 0.5)
        return amp, phs

def TensorHoloModel():
    
    return TH(BasicBlock, [2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    