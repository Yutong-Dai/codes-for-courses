import torch
import torch.nn as nn

#define the discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 196, 3, padding=1, stride=1),
                                   #196*32*32
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 32, 32]))
        self.conv2 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                  #196*16*16
                                   nn.LeakyReLU(),
                                   nn.LayerNorm([196, 16, 16])
                                  )
        self.conv3 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   #196*16*16
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 16, 16]))
        self.conv4 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                   #196*8*8
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 8, 8]))
        self.conv5 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   #196*8*8
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 8, 8]))
        self.conv6 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   #196*8*8
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 8, 8]))
        self.conv7 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=1),
                                   #196*8*8
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 8, 8]))
        self.conv8 = nn.Sequential(nn.Conv2d(196, 196, 3, padding=1, stride=2),
                                   #196*4*4
                                  nn.LeakyReLU(),
                                  nn.LayerNorm([196, 4, 4]))
        self.maxpool = nn.MaxPool2d(4, padding=0, stride=4) #196*1*1
        self.fc1 = nn.Linear(in_features=196, out_features=1)
        self.fc10 = nn.Linear(in_features=196, out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool(x)
        x = x.view(-1, 196)
        fc1_output = self.fc1(x)
        fc10_output = self.fc10(x)
        return [fc1_output, fc10_output]

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(100, 196*4*4),
                                nn.BatchNorm1d(196*4*4))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                  #196*8*8
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv2 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                  #196*8*8
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv3 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                  #196*8*8
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv4 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                  #196*8*8
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                  #196*16*16
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv6 = nn.Sequential(nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1),
                                  #196*16*16
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv7 = nn.Sequential(nn.ConvTranspose2d(196, 196, 4, padding=1, stride=2),
                                  #196*32*32
                                  nn.ReLU(),
                                  nn.BatchNorm2d(196))
        self.conv8 = nn.Conv2d(196, 3, 3, padding=1, stride=1)
    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(100, 196, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        output = torch.tanh(x)
        return output
        
            
        