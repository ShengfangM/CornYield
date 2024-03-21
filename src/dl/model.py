import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
import torchvision.models as models
import torch.optim as optim

''' vision transformer for regression'''
class ViTRegression_V0(nn.Module):
    def __init__(self, num_in_channel, base_model_name = 'vit-base-patch16-224-in21k-finetuned-imagenet'):
        super(ViTRegression_V0, self).__init__()
        self.n_channel = num_in_channel
        # self.vit = VisionTransformer.from_pretrained(base_model_name, num_channels=num_in_channel)
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # self.vit = models.vit_b_16()

        # setup for two class classification
        num_ftrs = self.vit.heads[-1].in_features
        self.vit.heads[-1] = torch.nn.Linear(num_ftrs, 1)
        # method 2: add another layer to 
        
        self.conv1 = nn.Conv2d(num_in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
            
        # self.regression_head = nn.Linear(self.vit.head.in_features, 1)
    def forward(self, x):
        if self.n_channel:
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
        
        x = self.vit(x)
        # x = self.regression_head(x[:, 0])  # Only use the [CLS] token for regression
        return x


''' using resnet, MLP and SDPA'''

class ResNetFNNTranfomerBase(nn.Module):
    def __init__(self, in_channel, num_metadata, num_class, resnet_feature_extractor):
        super(ResNetFNNTranfomerBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        num_resnet_features = self.feature_extract.num_features
        self.fc_metadata = nn.Linear(num_metadata, num_resnet_features)
        self.fc_combined = nn.Linear(num_resnet_features + num_resnet_features, num_class)
        self.fc = nn.Linear(num_resnet_features, num_class)
        
        self.num_heads = 2

    def forward(self, img, metadata):
        x_resnet = self.feature_extract(img)
        x_metadata = F.relu(self.fc_metadata(metadata))
        # print(x_resnet.size(1))
        # print(x_metadata.shape)
        
        # num_heads = 8
        # heads_per_dim = 64
        batch_size = x_resnet.size(0)
        head_dim = 35
        # embed_dim = x_resnet.size(2)
        # head_dim = embed_dim // (self.num_heads * 3)
        query = x_metadata.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = x_resnet.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = x_resnet.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        # x_combined = torch.cat((x_resnet, x_metadata), dim=1)
        # x_combined = x_combined.view(x_combined.size(0), -1)
        # x_out = self.fc_combined(x_combined)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x_combined = F.scaled_dot_product_attention(query,key,value)
        x_combined = x_combined.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_out = self.fc(x_combined)
        return x_out
    
class ResNetFNNTranfomer_V01(ResNetFNNTranfomerBase):
    def __init__(self, in_channel, num_metadata, num_class, resnet_name='resnet34'):
        super(ResNetFNNTranfomer_V01, self).__init__(in_channel, num_metadata, num_class, 
                                                     ResNetFeatures_V01(in_channel, 0, resnet_name=resnet_name))

class ResNetFNNTranfomer_V2(ResNetFNNTranfomerBase):
    def __init__(self, in_channel, num_metadata, num_class, num_extra_feature = 6, resnet_name='resnet18'):
        super(ResNetFNNTranfomer_V2, self).__init__(in_channel, num_metadata, num_class, 
                                                    CustomerFeatureExtraction(in_channel, 0, num_prior_feature = num_extra_feature, pretrained_net_name=resnet_name))

# class EfficientNetRegression(nn.Module):
#     def __init__(self, input_channels=5, efficientnet_variant='efficientnet_v2_s'):
#         super(EfficientNetRegression, self).__init__()
        
#         # Load the EfficientNet model with pre-trained weights
#         efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

#         print(*list(efficientnet.children())[:-1])
        
#         # Modify the input layer to accept 5 channels
#         efficientnet.Conv2dNormActivation[0] = nn.Conv2d(input_channels, 24,
#                                            kernel_size=3, stride=2, padding=1, bias=False)
#         # , 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
#         # Replace the classifier with a custom one
#         efficientnet._fc = nn.Linear(efficientnet._fc.in_features, 1)

#         self.model = efficientnet

#     def forward(self, x):
#         return self.model(x)


''' Only use the first layer of Resnet18 to extract feature'''
class CustomerFeatureExtraction(nn.Module):
    def __init__(self, in_channel, num_classes, num_prior_feature=6, pretrained_net_name = 'resnet18'):
        super(CustomerFeatureExtraction, self).__init__()
        
        self.features = ResnetFeatureSubset(in_channel, num_classes)
        # self.features = nn.Sequential(*list(resnet.children())[:-1])
        # # Define the fully connected layers
        self.fc1 = nn.Linear(in_channel, num_prior_feature - in_channel)  #
        # Define the spatial pooling layers
        self.pool0 = nn.AdaptiveAvgPool2d((2, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.num_features = 64 * 1+num_prior_feature
        self.in_channel = in_channel
        self.num_prior_feature = num_prior_feature
        
        
    def forward(self, x):
        
        # Apply convolutional layers
        x1 = self.pool1(x)
        # if self.in_channel < self.standard_channel:
        #     num_channels_to_replicate = num_output - in_channel
        #     # print(num_channels_to_replicate)
        #     x1_repeat = x1[:,-num_channels_to_replicate:,:,:]
        #     # print(x1_repeat.size())
        #     # print(x1.size())
        #     x1 = torch.cat((x1, x1_repeat), dim=1)
        x1 = x1.view(x1.size(0), -1)
        if self.num_prior_feature > self.in_channel:
            x1_2 = F.relu(self.fc1(x1))
            x1 = torch.cat((x1, x1_2), dim=1)
        
        x2 = self.features(x)

        # Concatenate the pooled features
        x = torch.cat((x1, x2), dim=1)
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        return x
    
    
    
class ResnetFeatureSubset(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained_net_name = 'resnet18'):
        super(ResnetFeatureSubset, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.resnet = models.resnet18(pretrained=True)
        if in_channel>3:
            weight = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = weight
                for ii in range(3, in_channel):
                    resnet.conv1.weight[:, ii] = resnet.conv1.weight[:, 2]
        self.features = resnet
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.num_features = 64
        # self.in_channel = in_channel
        
    def forward(self, x):
        
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)
        # Apply spatial pyramid pooling
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        return x
    
class SpatialPyramidCNN_V1(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(SpatialPyramidCNN_V1, self).__init__()
        
        self.features = ResnetFeatureSubset(in_channel, 0)
        self.fc1 = nn.Linear(self.features.num_features, 1)
        
    def forward(self, x):
        # Apply convolutional layers
        x1 = self.features(x)
        return self.fc1(x1)
    
class SpatialPyramidCNN_V2(nn.Module):
    def __init__(self, in_channel, num_classes, num_prior_feature = 6):
        super(SpatialPyramidCNN_V2, self).__init__()
        
        self.features = CustomerFeatureExtraction(in_channel, 0, num_prior_feature=num_prior_feature)
        self.fc1 = nn.Linear(self.features.num_features, 1)
        
    def forward(self, x):
        # Apply convolutional layers
        x1 = self.features(x)
        return self.fc1(x1)
        
'''modified ResNet model for regression
'''
# Base class for modified ResNet model for regression
class ResNetRegressionBase(nn.Module):
    def __init__(self, in_channel, num_classes, resnet_feature_extractor):
        super(ResNetRegressionBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        self.fc = nn.Linear(self.feature_extract.num_features, num_classes)

    def forward(self, img):
        x = self.feature_extract(img)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Subclasses with different ResNet feature extractors
class ResNetRegression_V00(ResNetRegressionBase):
    def __init__(self, in_channel, num_classes, resnet_name='resnet34'):
        super(ResNetRegression_V00, self).__init__(in_channel, num_classes, ResNetFeatures_V00(in_channel, 0, resnet_name=resnet_name))

class ResNetRegression_V01(ResNetRegressionBase):
    def __init__(self, in_channel, num_classes, resnet_name='resnet34'):
        super(ResNetRegression_V01, self).__init__(in_channel, num_classes, ResNetFeatures_V01(in_channel, 0, resnet_name=resnet_name))

class ResNetRegression_V10(ResNetRegressionBase):
    def __init__(self, in_channel, num_classes, resnet_name='resnet34'):
        super(ResNetRegression_V10, self).__init__(in_channel, num_classes, ResNetFeatures_V10(in_channel, 0, resnet_name=resnet_name))

class ResNetRegression_V11(ResNetRegressionBase):
    def __init__(self, in_channel, num_classes, resnet_name='resnet34'):
        super(ResNetRegression_V11, self).__init__(in_channel, num_classes, ResNetFeatures_V11(in_channel, 0, resnet_name=resnet_name))    



'''modified ResNet model and MLP for regression from image and metadata
'''
# Base class for combining ResNet and MLP for image and metadata
class ResNetFNNBase(nn.Module):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_feature_extractor):
        super(ResNetFNNBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        num_resnet_features = self.feature_extract.num_features
        # num_fc_features = num_resnet_features // 2
        num_fc_features = 64
        self.fc_metadata = nn.Linear(num_metadata, num_fc_features)
        self.fc_combined = nn.Linear(num_resnet_features + num_fc_features, num_classes)

    def forward(self, img, metadata):
        x_resnet = self.feature_extract(img)
        x_metadata = F.relu(self.fc_metadata(metadata))
        x_combined = torch.cat((x_resnet, x_metadata), dim=1)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_out = self.fc_combined(x_combined)
        return x_out

# Subclasses with different ResNet feature extractors
class ResNetFNN_V00(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_name='resnet34'):
        super(ResNetFNN_V00, self).__init__(in_channel, num_metadata, num_classes, ResNetFeatures_V00(in_channel, 0, resnet_name=resnet_name))

class ResNetFNN_V01(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_name='resnet34'):
        super(ResNetFNN_V01, self).__init__(in_channel, num_metadata, num_classes, ResNetFeatures_V01(in_channel, 0, resnet_name=resnet_name))

class ResNetFNN_V10(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_name='resnet34'):
        super(ResNetFNN_V10, self).__init__(in_channel, num_metadata, num_classes, ResNetFeatures_V10(in_channel, 0, resnet_name=resnet_name))

class ResNetFNN_V11(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_name='resnet34'):
        super(ResNetFNN_V11, self).__init__(in_channel, num_metadata, num_classes, ResNetFeatures_V11(in_channel, 0, resnet_name=resnet_name))
class ResNetFNN_V2(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, num_prior_feature = 1, resnet_name='resnet34'):
        super(ResNetFNN_V2, self).__init__(in_channel, num_metadata, num_classes, 
                                           CustomerFeatureExtraction(in_channel, 0, num_prior_feature= num_prior_feature, pretrained_net_name=resnet_name))
class ResNetFNN_V1(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_classes, resnet_name='resnet34'):
        super(ResNetFNN_V1, self).__init__(in_channel, num_metadata, num_classes, 
                                           ResnetFeatureSubset(in_channel, 0, pretrained_net_name=resnet_name))



'''# extract features from CNN models''' 
class ResNetFeatures_V00(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V00, self).__init__()\
        
        if resnet_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # method 0: change the first conv1 according to the in_channel
        if in_channel>3:
            weight = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
            
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = weight
                for ii in range(3, in_channel):
                    resnet.conv1.weight[:, ii] = resnet.conv1.weight[:, 2]

        self.num_features = num_features if num_features>0 else resnet.fc.in_features 
       
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.resnet.fc(x)
        return x
    

# modified ResNet model for regression
class ResNetFeatures_V01(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V01, self).__init__()
        
        if resnet_name == 'resnet18':
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # self.resnet = models.resnet18(pretrained=True)
        if in_channel>3:
            self.weight = self.resnet.conv1.weight.clone()
            self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = self.weight
                for ii in range(3, in_channel):
                    self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]

        self.num_features = num_features if num_features>0 else self.resnet.fc.in_features 
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


# extract features from CNN models by adding a conv layer to
class ResNetFeatures_V10(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V10, self).__init__()

        if resnet_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
        self.num_features = num_features if num_features>0 else self.resnet.fc.in_features 
    def forward(self, x):
        
        if self.in_channel >3:
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return x

# modified ResNet34 model for regression
class ResNetFeatures_V11(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V11, self).__init__()
        
        self.in_channel = in_channel

        if resnet_name == 'resnet18':
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)


        self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        # self.resnet.fc = nn.Linear(512, 1)
        self.num_features = num_features if num_features>0 else self.resnet.fc.in_features 
        
    def forward(self, x):

        if self.in_channel >3:
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
        
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
    
''' self build'''    
# modified ResNet34 model for regression
class CustomModel(nn.Module):
    def __init__(self, in_channel, num_metadata, num_features, resnet_name : str = 'resnet34'):
        super(CustomModel, self).__init__()

        # self.weight = self.resnet.conv1.weight.clone()
        # self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
        # with torch.no_grad():
        #     self.resnet.conv1.weight[:, :3] = self.weight
        #     for ii in range(3, in_channel):
        #         self.resnet.conv1.weight[:, ii] = self.resnet.conv1.weight[:, 2]
        self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        # self.resnet = ......

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
        self.conv = ResNetRegression_V01(self.input_channels, self.num_features)
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
        # self.conv1 = nn.Conv2d(in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channel,32, kernel_size=3, stride=1, padding= 0, bias=False)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5, stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64,128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(3, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(64*6, 128)
        # self.fc2 = nn.Linear(4098, 1028)
        self.fc3 = nn.Linear(128, 1)
        # Define the spatial pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        # print(x.size(1),  x.size(2), x.size(3))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # print(x.size())
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class SpatialPyramidCNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(SpatialPyramidCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 9+in_channel, 1)  # 13x13 is the output size after spatial pooling
        
        # Define the spatial pooling layers
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 4))
        # self.pool3 = nn.AdaptiveMaxPool2d((2, 4))
        
    def forward(self, x):
        # Apply convolutional layers
        x1 = self.pool1(x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Apply spatial pyramid pooling
        x2 = self.pool1(x)
        x3 = self.pool2(x)
        
        # Flatten the features
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        # Concatenate the pooled features
        x = torch.cat((x1, x2, x3), dim=1)
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        return self.fc1(x)

        
# class SpatialPyramidCNN_V2(nn.Module):
#     def __init__(self, in_channel, num_classes):
#         super(SpatialPyramidCNN_V2, self).__init__()
        
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         # self.resnet = models.resnet18(pretrained=True)
#         if in_channel>3:
#             weight = resnet.conv1.weight.clone()
#             resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
#             with torch.no_grad():
#                 resnet.conv1.weight[:, :3] = weight
#                 for ii in range(3, in_channel):
#                     resnet.conv1.weight[:, ii] = resnet.conv1.weight[:, 2]
#         self.features = resnet
#         # self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        
#         # Define the fully connected layers
#         self.fc1 = nn.Linear(64 * 1+in_channel, 1)  # 13x13 is the output size after spatial pooling
        
#         # Define the spatial pooling layers
#         self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
#         # self.pool2 = nn.AdaptiveAvgPool2d((2, 4))
#         # self.pool3 = nn.AdaptiveMaxPool2d((2, 4))
        
#     def forward(self, x):
#         # Apply convolutional layers
#         x1 = self.pool1(x)
        
#         x = self.features.conv1(x)
#         x = self.features.bn1(x)
#         x = self.features.relu(x)
#         x = self.features.maxpool(x)

#         x = self.features.layer1(x)
        
#         # Apply spatial pyramid pooling
#         x2 = self.pool1(x)
#         # x3 = self.pool2(x)
        
#         # Flatten the features
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         # x3 = x3.view(x3.size(0), -1)
        
#         # Concatenate the pooled features
#         x = torch.cat((x1, x2), dim=1)
#         # Flatten the features
#         x = x.view(x.size(0), -1)
        
#         return self.fc1(x)
    
'''NN regression'''
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size=0, num_classes=0):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        
    def forward(self, x):
        x = torch.flatten(x,1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
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
    
    