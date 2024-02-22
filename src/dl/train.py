from sklearn.model_selection import KFold
import torch
from torchvision import transforms
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_cross_validation(model, train_val_dataset:torch.utils.data.Dataset, batch_size, num_epochs, optimizer, criterion,
                                n_split: int = 4, shuffle: bool=True, random_state: int = 39, dual_model:bool=False, model2=None, is_dual_data = False):
    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # kf = KFold(n_splits=4, shuffle=False)
    kf = KFold(n_splits=4, shuffle=shuffle, random_state=random_state)

    val_accuracy_all =[]

    for train_indices, val_indices in kf.split(train_val_dataset):
        train_dataset = torch.utils.data.Subset(train_val_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(train_val_dataset, val_indices)
        if dual_model:
            model, model2 = train_dual_models(model, model2, train_dataset, batch_size, num_epochs, optimizer, criterion)
            val_accuracy = validate_dual_models(model, model2, val_dataset,criterion, batch_size)
        else:
            model = train(model, train_dataset, batch_size, num_epochs, optimizer, criterion,is_dual_data = is_dual_data)
            val_accuracy = validate(model, val_dataset,criterion, batch_size,is_dual_data = is_dual_data)
        val_accuracy_all.append(val_accuracy)
        print(f'validation mse is {np.mean(val_accuracy_all)}')
        print(f'All validation mse is {np.mean(val_accuracy)}')

    return model, val_accuracy_all




def train(model, train_dataset, batch_size, num_epochs, optimizer, criterion, is_dual_data = False):  
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
        

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train your model on the training data
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            # Get inputs and labels
            inputs, labels = data
            # labels = labels.unsqueeze(1) #change to (32,1)
            labels = labels.view(-1, 1)
            # Move batch of images and captions to GPU if CUDA is available.

            if is_dual_data:
                img, metadata = inputs
                img = img.to(DEVICE)
                metadata = metadata.to(DEVICE)
                labels = labels.to(DEVICE)
                    
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(img, metadata)
            else:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
            
            # Compute loss
            if  outputs.shape == labels.shape:
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                # update weights
                optimizer.step()
            else:
                print('outputs shape is ', outputs.shape, ' the size is ', outputs.size())
                print('labels shape is ', labels.shape, ' the size is ', labels.size())
                
                # # Print statistics
                # running_loss += loss.item()
                # train_loss +=  loss.item()
                # if i % 100 == 99:    # Print every 100 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #         (epoch + 1, i + 1, running_loss))
                #     running_loss = 0.0
    return model


''' validate the model
return the output when is_return_output is true
'''
def validate(model, val_dataset,criterion, batch_size : int = 100, shuffle: bool=False, is_return_output: bool = False, is_dual_data = False):

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    # Evaluate the model on the validation data
    model.eval()
    val_accuracy = []
    if is_return_output:  
        test_pred = [0]*(len(val_dataset))

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        labels = labels.view(-1, 1)
        if is_dual_data:
                img, metadata = inputs
                img = img.to(DEVICE)
                metadata = metadata.to(DEVICE)
                labels = labels.to(DEVICE)
                    
                # Forward pass
                outputs = model(img, metadata)
        else:
            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)


        loss = criterion(outputs, labels)
        # print(outputs.shape, labels.shape)
        val_accuracy.append(float(loss.mean()))

        if is_return_output:  
            last_position = min(i*batch_size+batch_size, len(test_pred))
            test_pred[i*batch_size:last_position] = outputs.cpu().detach().numpy().flatten()
    
    if is_return_output:  
        return val_accuracy, test_pred
    else:
        return val_accuracy


def data_transform():

    return transforms.Compose([
        # NumpyArrayToTensor(),  # Custom transformation
        # transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225, 0.225]),  # Example normalization values
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization values

    ])


def data_transform_vit():

    return transforms.Compose([
        # NumpyArrayToTensor(),  # Custom transformation
        # transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225, 0.225]),  # Example normalization values
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization values

    ])


def data_resize():

    return transforms.Compose([
        # NumpyArrayToTensor(),  # Custom transformation
        # transforms.ToTensor(),
        transforms.Resize((224,224))
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=30),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225, 0.225]),  # Example normalization values
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization values

    ])

def train_dual_models(model1, model2, train_dataset, batch_size, num_epochs, optimizer, criterion):          

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train your model on the training data
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            # Get inputs and labels
            inputs, labels = data
            # labels = labels.unsqueeze(1) #change to (32,1)
            labels = labels.view(-1, 1)
            # Move batch of images and captions to GPU if CUDA is available.
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            features = model1(inputs)
            outputs = model2(features)
            
            # Compute loss
            if  outputs.shape == labels.shape:
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                # update weights
                optimizer.step()
            else:
                print('outputs shape is ', outputs.shape, ' the size is ', outputs.size())
                print('labels shape is ', labels.shape, ' the size is ', labels.size())
                
                # # Print statistics
                # running_loss += loss.item()
                # train_loss +=  loss.item()
                # if i % 100 == 99:    # Print every 100 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #         (epoch + 1, i + 1, running_loss))
                #     running_loss = 0.0
    return model1, model2


''' validate the model
return the output when is_return_output is true
'''
def validate_dual_models(model1,model2, val_dataset,criterion, batch_size : int = 100, shuffle: bool=False, is_return_output: bool = False):

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    # Evaluate the model on the validation data
    model1.eval()
    model2.eval()
    val_accuracy = []
    if is_return_output:  
        test_pred = [0]*(len(val_dataset))

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        labels = labels.view(-1, 1)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        features = model1(inputs)
        outputs = model2(features)

        loss = criterion(outputs, labels)
        # print(outputs.shape, labels.shape)
        val_accuracy.append(float(loss.mean()))

        if is_return_output:  
            last_position = min(i*batch_size+batch_size, len(test_pred))
            test_pred[i*batch_size:last_position] = outputs.cpu().detach().numpy().flatten()
    
    if is_return_output:  
        return val_accuracy, test_pred
    else:
        return val_accuracy
