import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim


class ConvEncoder(nn.Module):
    """ create convolutional layers to extract features
    from input multipe spectral images

    Attributes:
    data : input data to be encoded
    """

    def __init__(self, in_channel):
        super(ConvEncoder,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=64, kernel_size=4,stride=1, padding=0)
        # nn.init.xavier_uniform(self.conv2.weight)
        self.relu= nn.ReLU()

        #Max Pool 1
        # self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        # nn.init.xavier_uniform(self.conv2.weight)
        # self.swish2 = nn.ReLU()

        #Max Pool 2
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        # nn.init.xavier_uniform(self.conv3.weight)
        # self.relu = nn.ReLU()

    def forward(self,x):
        out=self.conv1(x)
        out=self.relu(out)
        size1 = out.size()
        # out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.relu(out)
        size2 = out.size()
        # out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.relu(out)
        return(out)



class DeConvDecoder(nn.Module):
    """ 
    reconstruct image from extracted features

    Attributes:
    features : input data to be encoded
    in_channel: reconstructed channels
    """
    def __init__(self, in_channel):
        super(DeConvDecoder,self).__init__()

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=3)
        # nn.init.xavier_uniform(self.deconv1.weight)
        # self.swish4=nn.ReLU()
        #Max UnPool 1
        # self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=5)
        # nn.init.xavier_uniform(self.deconv2.weight)
        # self.swish5=nn.ReLU()

        #Max UnPool 2
        # self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=64,out_channels=in_channel,kernel_size=4)
        # nn.init.xavier_uniform(self.deconv3.weight)
        # self.swish6=nn.ReLU()
        self.relu= nn.ReLU()

    def forward(self,x):
        out=self.deconv1(x)
        out=self.relu(out)
        # out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.relu(out)
        # out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        # out=self.swish6(out)
        return(out)
    
    
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size=0, num_classes=0):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16*input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16*input_size, input_size)
        self.fc3 = nn.Linear(input_size, 4096)
        self.fc4 = nn.Linear(4096, 512)
        self.fc5 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = torch.flatten(x,1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out
    
    
# Define the modified ResNet34 model
class ResNet34_R(nn.Module):
    def __init__(self, in_channel):
        super(ResNet34_R, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = self.weight
            for ii in range(3, in_channel):
                self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]

        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        return x