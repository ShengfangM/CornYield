import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim

from dataset import select_data_and_yield_list
from ml.ml_predict import plot_result_separate
from dl.dl_dataset import CornDataset
from dl.model import ResNetRegression
from dl.train import train_with_cross_validation, train, validate, data_transform


def data_prepare(yield_file, img_path, out_path, keyword, selection):

    if not os.path.exists(out_path):
        # If it doesn't exist, create the directory and any missing parent directories
        os.makedirs(out_path)
        print(f"Directory '{out_path}' created successfully.")
    else:
        print(f"Directory '{out_path}' already exists.")

    pioneer_img_list, pioneer_yield_list, irrigate_type_list = select_data_and_yield_list(
        img_path, yield_file, key_word = keyword, crop_type_select=selection)
    
    # for suffix_list in suffix_list_list:
    total_size = len(pioneer_yield_list)
    train_size = int(0.8 * total_size)  # 80% for training
    test_size = int(0.2 * total_size)   # 20% for validation
 
    # Use train_test_split to split the indices into training and testing sets
    train_indices, test_indices = train_test_split(range(len(pioneer_img_list)), test_size=test_size, random_state=39)

    train_val_dataset = CornDataset([pioneer_img_list[i] for i in train_indices], [pioneer_yield_list[i] for i in train_indices], transform=data_transform())
    test_dataset = CornDataset([pioneer_img_list[i] for i in test_indices], [pioneer_yield_list[i] for i in test_indices])

    return pioneer_yield_list,train_val_dataset, test_dataset, irrigate_type_list, test_indices


def predict_yield_from_img(yield_file, img_path, out_path, is_save_model, is_test):
    
    selection = ['Pioneer'] # 
    # selection = 'Pioneer Deficit' 
    # selection = 'Pioneer Full'

    # key_word_list = ['Ref_filled.tif', 'RGB_filled.tif']
    key_word_list = ['Ref_filled.tif']
    suffix_list_list = [[], ['LWIR_filled.tif']]

    # suffix = ['base', 'lwir']
    # suffix_list = ['LWIR_filled.tif']
    # VI_list = ['ndvi', 'ndre', 'gndvi', 'evi']
    VI_list = ['evi']
    for keyword in key_word_list:

        pioneer_yield_list, train_val_dataset, test_dataset, irrigate_type_list, test_indices= data_prepare(yield_file, img_path, out_path, keyword, selection)
        # yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
        doy_name = img_path[-23:-17]

                
        in_channel = 5
        num_epochs = 200
        batch_size = 32

        # Initialize an empty list to store fold-wise performance
        fold_accuracies = []

        # Initialize a new model for each fold
        # model = CNNRegression(in_channel)
        model = ResNetRegression(in_channel, 1)
        # model = EncoderCNN(in_channel, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)


        criterion = torch.nn.MSELoss()  # Mean Squared Error loss function
        # optimizer = optim.Adam(list(conv.parameters()) + list(deconv.parameters()), lr=0.001)  # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        # optimizer.zero_grad()
        model = train_with_cross_validation(model, train_val_dataset, batch_size, num_epochs, optimizer, criterion)

        if is_save_model:
            model_name = "path/model_" + model.__class__.__name__+ doy_name +"_Batch=" +str(batch_size) + "_state.pth"
            torch.save(model.state_dict(), model_name)

        test_accuracy, test_prediction = validate(model, test_dataset, criterion, batch_size = batch_size, is_return_output = True)

        print(f'validation mse is {np.sqrt(np.mean(test_accuracy))}')


        name_tag = doy_name
        out_name = name_tag + ' ' + 'Pioneer' + ' ' 
        out_name = out_name + keyword[:-11] 
        out_name = out_name + ' CNN '

        yield_data = np.array(pioneer_yield_list)
        irrigate_data = np.array(irrigate_type_list)

        test_truth = yield_data[test_indices]
        plot_result_separate(np.array(test_truth), np.array(test_prediction), test_indices, irrigate_data, out_name)


def predict_yield(yield_file, img_path, out_path, is_save_model, is_test):
    
    selection = ['Pioneer'] # 
    selection = None # 
    # selection = 'Pioneer Deficit' 
    # selection = 'Pioneer Full'

    # key_word_list = ['Ref_filled.tif', 'RGB_filled.tif']
    key_word_list = ['Ref_filled.tif']
    suffix_list_list = [[], ['LWIR_filled.tif']]

    # suffix = ['base', 'lwir']
    # suffix_list = ['LWIR_filled.tif']
    # VI_list = ['ndvi', 'ndre', 'gndvi', 'evi']
    VI_list = ['evi']
    for keyword in key_word_list:

        pioneer_yield_list, train_val_dataset, test_dataset, irrigate_type_list, test_indices= data_prepare(yield_file, img_path, out_path, keyword, selection)
        # yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
        doy_name = img_path[-23:-17]

                
        in_channel = 5
        num_epochs = 200
        batch_size = 32

        # Initialize an empty list to store fold-wise performance
        fold_accuracies = []

        # Initialize a new model for each fold
        # model = CNNRegression(in_channel)
        model = ResNetRegression(in_channel, 1)
        # model = EncoderCNN(in_channel, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)


        criterion = torch.nn.MSELoss()  # Mean Squared Error loss function
        # optimizer = optim.Adam(list(conv.parameters()) + list(deconv.parameters()), lr=0.001)  # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        # optimizer.zero_grad()
        model = train_with_cross_validation(model, train_val_dataset, batch_size, num_epochs, optimizer, criterion)

        if is_save_model:
            model_name = "path/model_" + model.__class__.__name__+ doy_name +"_Batch=" +str(batch_size) + "_state.pth"
            torch.save(model.state_dict(), model_name)

        test_accuracy, test_prediction = validate(model, test_dataset, criterion, batch_size = batch_size, is_return_output = True)

        print(f'validation mse is {np.sqrt(np.mean(test_accuracy))}')


        name_tag = doy_name
        out_name = name_tag + ' ' + 'ALL' + ' ' 
        out_name = out_name + keyword[:-11] 
        out_name = out_name + ' CNN'

        yield_data = np.array(pioneer_yield_list)
        irrigate_data = np.array(irrigate_type_list)

        test_truth = yield_data[test_indices]
        plot_result_separate(np.array(test_truth), np.array(test_prediction), test_indices, irrigate_data, out_name)


if __name__== "__main__":
    yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
    out_path = '../output/'
    img_root_path = 'D:/Corn_Yield/UAV_Data_Extracted_filled/'
    path_list = ['LIRF20220628_DOY179_extracted_filled', 'LIRF20220708_DOY189_extracted_filled', 'LIRF20220716_DOY197_extracted_filled'\
                 'LIRF20220720_DOY201_extracted_filled', 'LIRF20220916_DOY259_extracted_filled', 'LIRF20220926_DOY269_extracted_filled']
    
    for path_i in path_list:
        img_path = img_root_path+path_i

        predict_yield_from_img(yield_file, img_path, out_path, True, True)


    