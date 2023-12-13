import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim


# modified ResNet34 model for regression
class ResNetFeatures(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures, self).__init__()
        if resnet_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet34(pretrained=True)

        # self.resnet = models.resnet18(pretrained=True)
        self.weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = self.weight
            for ii in range(3, in_channel):
                self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]

        # self.resnet.fc = nn.Linear(512, 1)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)

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
        # x = self.resnet.fc(x)
        return x


class NNRegression(nn.Module):
    def __init__(self, num_features):
        super(NNRegression, self).__init__()        
        # self.resnet.fc = nn.Linear(512, 1)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):        
        x = self.fc(x)
        return x    



# modified ResNet34 model for regression
class ResNetRegression(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetRegression, self).__init__()
        if resnet_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet34(pretrained=True)

        # self.resnet = models.resnet18(pretrained=True)
        self.weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = self.weight
            for ii in range(3, in_channel):
                self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]

        # self.resnet.fc = nn.Linear(512, 1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)

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
            
            
# # version 1
# # modified ResNet34 model for regression
# class ResNetFNN(nn.Module):
#     def __init__(self, in_channel, num_metadata, num_features, resnet_name : str = 'resnet34'):
#         super(ResNetFNN, self).__init__()
#         if resnet_name == 'resnet18':
#             self.resnet = models.resnet18(pretrained=True)
#         elif resnet_name == 'resnet50':
#             self.resnet = models.resnet50(pretrained=True)
#         else:
#             self.resnet = models.resnet34(pretrained=True)

#         self.weight = self.resnet.conv1.weight.clone()
#         self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
#         with torch.no_grad():
#             self.resnet.conv1.weight[:, :3] = self.weight
#             for ii in range(3, in_channel):
#                 self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]
#         # self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 

#         # self.resnet.fc = nn.Linear(512, 1)
#         # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features1)
#         num_features2 = 64
#         self.fc = nn.Linear(num_metadata, num_features2)

#         self.fc2 = nn.Linear(self.resnet.fc.in_features+num_features2, num_features)

#     def forward(self, img, metadata):
#         # x = self.conv1(img)
#         x = self.resnet.conv1(img)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)

#         x = self.resnet.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # x = self.resnet.fc(x)

#         x2 = self.fc(metadata)

#         x = torch.cat((x, x2), dim=1)
#         x = x.view(x.size(0), -1)

#         x = self.fc2(x)

#         return x
    
# version 2            
# modified ResNet34 model for regression
class ResNetFNN(nn.Module):
    def __init__(self, in_channel, num_metadata, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFNN, self).__init__()
        if resnet_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet34(pretrained=True)

        # self.weight = self.resnet.conv1.weight.clone()
        # self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
        # with torch.no_grad():
        #     self.resnet.conv1.weight[:, :3] = self.weight
        #     for ii in range(3, in_channel):
        #         self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]
        self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 

        # self.resnet.fc = nn.Linear(512, 1)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features1)
        num_features2 = 64
        self.fc = nn.Linear(num_metadata, num_features2)

        self.fc2 = nn.Linear(self.resnet.fc.in_features+num_features2, num_features)

    def forward(self, img, metadata):
        x = self.conv1(img)
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
        # x = self.resnet.fc(x)

        x2 = self.fc(metadata)

        x = torch.cat((x, x2), dim=1)
        x = x.view(x.size(0), -1)

        x = self.fc2(x)

        return x
            
# '''feature extractor'''
# class EncoderCNN(nn.Module):
#     def __init__(self, in_channel, num_features, resnet_name = 'resnet34'):
#         super(EncoderCNN, self).__init__()
        
#         if resnet_name == 'resnet18':
#             resnet = models.resnet18(pretrained=True)
#         elif resnet_name == 'resnet50':
#             resnet = models.resnet50(pretrained=True)
#         else:
#             resnet = models.resnet34(pretrained=True)

#         weight = resnet.conv1.weight.clone()
#         resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
#         with torch.no_grad():
#             resnet.conv1.weight[:, :3] = weight
#             for ii in range(3, in_channel):
#                 resnet.conv1.weight[:, ii] = resnet.conv1.weight[:, 2]
                
#         # for param in resnet.parameters():
#         #     param.requires_grad_(False)
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         # self.fc = nn.Linear(512, num_features)
#         # print(resnet.fc.in_features, num_features)
#         self.fc = nn.Linear(resnet.fc.in_features, num_features)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.fc(features)
#         return features


class ConvLSTMRegression(nn.Module):
    def __init__(self, input_channels, num_features, hidden_channels, n_time_step, kernel_size, num_layers: int = 1):
        super(ConvLSTMRegression, self).__init__()

        self.input_channels = input_channels
        self.num_features = num_features
        self.hidden_size = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a list of ConvLSTM layers
        self.conv = ResNetRegression(self.input_channels, self.num_features)
        self.lstm = nn.LSTMCell(self.num_features, self.hidden_size)
        self.conv2 = nn.Conv2d(n_time_step, 1, kernel_size=1, bias=False)
        self.fc = nn.Linear(self.hidden_size, 1)

        
    #     self.lstm = nn.ModuleList()
    #     for _ in range(num_layers):
    #         self.lstm.append(self._make_layer())

    # def _make_layer(self):

    #     conv = EncoderCNN(self.input_channels, self.num_features)
    #     # Define a single ConvLSTM layer
    #     return nn.Sequential(
    #         nn.LSTMCell(self.hidden_channels, self.hidden_channels),
    #     )                     

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros(( batch_size, self.hidden_size), device=self.device), \
                torch.zeros(( batch_size, self.hidden_size), device=self.device))
               
    def forward(self, input_sequence):
        """ Define the feedforward behavior of the model """
        # Initialize the hidden state
        batch_size, seq_len, _, _, _ = input_sequence.size() #features is of shape (batch_size, embed_size)
#         print(f'batch_size: {self.batch_size}')
        hidden_states, cell_states = self.init_hidden(batch_size) 
        # print(hidden_states.shape, cell_states.shape)
        # cell_states = self.init_hidden(self.batch_size) 

        output = []
        for t in range(seq_len):
            x = input_sequence[:, t, :, :, :]
            # for i in range(self.num_layers):
            x= self.conv(x)
            hidden_states, cell_states = self.lstm(x, (hidden_states, cell_states)) 
            output.append(hidden_states)
        
        # x = self.conv2(output)
        result = self.fc(hidden_states) # outputs shape : (batch_size, MAX_LABEL_LEN, vocab_size)
#         print(f'outputs: {outputs.shape}')
        return result
        

'''in'''

class LSTMRegression(nn.Module):
    def __init__(self, embed_size, hidden_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )

        self.linear = nn.Linear(hidden_size, 1)                     

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=self.device), \
                torch.zeros((1, batch_size, self.hidden_size), device=self.device))
               
    def forward(self, features):
        """ Define the feedforward behavior of the model """
        # Initialize the hidden state
        self.batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
#         print(f'batch_size: {self.batch_size}')
        self.hidden = self.init_hidden(self.batch_size) 
        lstm_out, self.hidden = self.lstm(features, self.hidden) # lstm_out shape : (batch_size, MAX_LABEL_LEN, hidden_size)

        # lstm_out, self.hidden = self.lstm(features.unsqueeze(1), self.hidden) # lstm_out shape : (batch_size, MAX_LABEL_LEN, hidden_size)
        # print(f'lstm_out: {lstm_out.shape}')
#         print(f'hidden: {self.hidden[0].shape}')
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, MAX_LABEL_LEN, vocab_size)
#         print(f'outputs: {outputs.shape}')
        return outputs
    

'''self defined CNN regression model'''
class CNNRegression(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        #self.conv1 = nn.Conv2d(in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channel,64, kernel_size=(5,7), stride=1, padding=(2,3), bias=False)
        self.conv2 = nn.Conv2d(64,128, kernel_size=(3,7), stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128,256, kernel_size=(3,5), stride=1, padding=0, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*37*3, 4098)
        self.fc2 = nn.Linear(4098, 1028)
        self.fc3 = nn.Linear(1028, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        # print(x.size(1),  x.size(2), x.size(3))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

# modified ResNet34 model for regression
class ResNet34Regression(nn.Module):
    def __init__(self, in_channel):
        super(ResNet34Regression, self).__init__()
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
    
'''NN regression'''
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

'''encoder'''
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
    
    