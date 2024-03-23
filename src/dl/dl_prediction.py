import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim

from dataset import get_imgfilelist_yield, select_data_and_yield_list, create_metadata

from ml.ml_predict import plot_result_separate
from dl.dl_dataset import CornDataset, MixedDataset
from dl.model import (ResNetRegression_V00, ResNetRegression_V10, ViTRegression_V0, 
                    ResNetRegression_V01, ResNetFNN_V00, ResNetFNN_V01,ResNetFNN_V2, ResNetFNN_V1,
                    ResNetFNNTranfomer_V01, ResNetFNNTranfomer_V2,
                    SpatialPyramidCNN, SpatialPyramidCNN_V1, SpatialPyramidCNN_V2)
from dl.train import train_with_cross_validation, train, validate, data_transform, data_resize, data_transform_vit
from plot_utils import plot_distinct_yields

import time

crop_var={
    3: 'all',
    2: 'Pioneer',
    1: 'CH 192-10',
    0: 'DKC 51-91'
}

irrigate_var={
    2: ' ',
    0: 'Full',
    1: 'Deficit'
}

def select_imglist_yield(yield_file, img_path, keyword, analyze_variety_id=3, analyze_irrigation_id=2):
    '''get selected image files and yield data according to corn variety and irrigate type'''

    img_list, yield_pf = get_imgfilelist_yield(img_path, yield_file, keyword)

    if analyze_variety_id != 3 or analyze_irrigation_id != 2:
        if analyze_variety_id != 3 and analyze_irrigation_id != 2:
            yield_pf = yield_pf[(yield_pf['Variety_int'] == analyze_variety_id) & (yield_pf['Irrigation_int'] == analyze_irrigation_id)]
        elif analyze_variety_id != 3:
            yield_pf = yield_pf[yield_pf['Variety_int'] == analyze_variety_id]
        elif  analyze_irrigation_id != 2:
            yield_pf = yield_pf[ yield_pf['Irrigation_int'] == analyze_irrigation_id] 

        indices = yield_pf.index.tolist()
        img_list = [img_list[i] for i in indices]
        yield_pf = yield_pf.reset_index(drop=True)

    return img_list, yield_pf


def get_train_test_img(img_list, yield_pf, train_col='TRAIN_75', VI_list=None, suffix_list = None):

    yield_list = list(yield_pf['Yield_Bu_Ac'])
    train_indices = list(yield_pf[yield_pf[train_col] == 1].index)
    test_indices = list(yield_pf[yield_pf[train_col] == 0].index)

    # ### vit
    # train_val_dataset = CornDataset([img_list[i] for i in train_indices], [yield_list[i] for i in train_indices], transform=data_transform_vit())
    # test_dataset = CornDataset([img_list[i] for i in test_indices], [yield_list[i] for i in test_indices], transform=data_resize())

    train_val_dataset = CornDataset([img_list[i] for i in train_indices], [yield_list[i] for i in train_indices], transform=data_transform())
    test_dataset = CornDataset([img_list[i] for i in test_indices], [yield_list[i] for i in test_indices])

    return yield_list,train_val_dataset, test_dataset, test_indices


def get_train_test_img_metadata(img_list, yield_pf, doy, weather_file, train_col='TRAIN_75', 
                                VI_list=None, suffix_list = None):

    yield_list = list(yield_pf['Yield_Bu_Ac'])
    train_indices = list(yield_pf[yield_pf[train_col] == 1].index)
    test_indices = list(yield_pf[yield_pf[train_col] == 0].index)
    
    # doy=int(img_path[-20:-17])
    metadata = create_metadata(yield_pf, weather_file, doy)

    ### vit
    # train_val_dataset = MixedDataset([img_list[i] for i in train_indices], [yield_list[i] for i in train_indices], metadata.loc[train_indices], VI_list=VI_list, suffix_list=suffix_list,  transform=data_transform_vit())
    # test_dataset = MixedDataset([img_list[i] for i in test_indices], [yield_list[i] for i in test_indices], metadata.loc[test_indices], VI_list=VI_list, suffix_list=suffix_list,  transform=data_resize())


    train_val_dataset = MixedDataset([img_list[i] for i in train_indices], [yield_list[i] for i in train_indices], metadata.loc[train_indices], VI_list=VI_list, suffix_list=suffix_list,  transform=data_transform())
    test_dataset = MixedDataset([img_list[i] for i in test_indices], [yield_list[i] for i in test_indices], metadata.loc[test_indices], VI_list=VI_list, suffix_list=suffix_list)


    return yield_list,train_val_dataset, test_dataset, test_indices


def predict_yield_from_img(yield_file, img_path, out_path, is_save_model, is_test, analyze_variety_id=2, 
                           analyze_irrigation_id=2, key_word_list = ['Ref_filled.tif'], resname='resnet18'):
    
    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    suffix_list_list = [[], ['LWIR_filled.tif']]

    # suffix = ['base', 'lwir']
    # suffix_list = ['LWIR_filled.tif']
    # VI_list = ['ndvi', 'ndre', 'gndvi', 'evi']
    VI_list = None
    train_col='TRAIN_75'
    for keyword in key_word_list:

        img_list, yield_pf = select_imglist_yield(yield_file, img_path, keyword, 
                                                  analyze_variety_id=analyze_variety_id, 
                                                  analyze_irrigation_id=analyze_irrigation_id)

        pioneer_yield_list, train_val_dataset, test_dataset, test_indices= get_train_test_img(img_list, yield_pf, train_col=train_col,
                                                                                              VI_list=VI_list)

        sample_data, _ = train_val_dataset[0]
        print(sample_data.shape[0])

        # yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
        doy_name = img_path[-23:-17]

        in_channel = sample_data.shape[0]
        num_epochs = 120
        batch_size = 32

        # Initialize a new model for each fold
        # resname='resnet34'
        # model = CNNRegression(in_channel)
        # model = ResNetRegression_V10(in_channel, 1, resname)
        # model = ResNetRegression_V01(in_channel, 1, resname)
        # model = ResNetRegression_V00(in_channel, 1, resname)
        # model = ViTRegression_V0(in_channel)
        # model = EfficientNetRegression(in_channel)
        # model = SpatialPyramidCNN_V2(in_channel, 1, in_channel)
        # model = SpatialPyramidCNN_V2(in_channel, 1, num_prior_feature= in_channel)
        model = SpatialPyramidCNN_V1(in_channel, 1)

        # model = SpatialPyramidCNN(in_channel, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        criterion = torch.nn.MSELoss()  # Mean Squared Error loss function
        # optimizer = optim.Adam(list(conv.parameters()) + list(deconv.parameters()), lr=0.001)  # Adam optimizer
        lr = 0.00075
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

        # optimizer.zero_grad()
        cur_time = time.time()
        model, val_mse = train_with_cross_validation(model, train_val_dataset, batch_size, num_epochs, optimizer, criterion)
        training_time = time.time() - cur_time
        print()
        print('training time is  : ', training_time)
        print()

        if is_save_model:
            # model_name = "path/model_" + crop_var[analyze_variety_id] +"_img(nbands="+str(in_channel)+ ')_'+ str(doy_name) + "-"+model.__class__.__name__+'_'+resname+ "_Batch=" +str(batch_size) + "_Epoch=" +str(num_epochs) + "_lr=" +str(lr)+ "_state.pth"
            # torch.save(model.state_dict(), model_name)
            model_name = "path/model_" + crop_var[analyze_variety_id] +"_img(nbands="+str(in_channel)+ ')_'+ str(doy_name) + "-"+model.__class__.__name__+'_'+resname+ "_Batch=" +str(batch_size) + "_Epoch=" +str(num_epochs) + "_lr=" +str(lr)+ "_model.pkl"
            torch.save(model, model_name)

        cur_time = time.time()
        test_accuracy, test_prediction = validate(model, test_dataset, criterion, batch_size = batch_size, is_return_output = True)
        print(f'validation mse is {np.sqrt(np.mean(test_accuracy))}')
        test_time = time.time() - cur_time
        print()
        print('test time is  : ', test_time)
        print()

        name_tag = doy_name + ' ' + crop_var[analyze_variety_id] + ' ' + irrigate_var[analyze_irrigation_id] + ' '
        out_name = name_tag
        out_name = out_name + keyword[:-11] 
        out_name = out_name + model.__class__.__name__ +' Resnet18 '

        yield_data = np.array(pioneer_yield_list)
        test_irrigate_data = np.array(yield_pf[yield_pf[train_col] == 0]['Irrigation_int'])
        test_variety_data = np.array(yield_pf[yield_pf[train_col] == 0]['Variety_int'])

        test_truth = yield_data[test_indices]

        result_df=pd.DataFrame({
            'Truth': np.array(test_truth), 
            'Prediction':np.array(test_prediction),
            'Irrigation_int':test_irrigate_data,
            'Vriaty_int':test_variety_data
        })
        csv_file_path = out_path+out_name + '.csv'
        result_df.to_csv(csv_file_path, index=False)

        title = name_tag + keyword[:-11].upper() + ' CNN'
        plot_distinct_yields(np.array(test_truth)*0.06277, np.array(test_prediction)*0.06277, test_irrigate_data, test_variety_data, title, out_path+out_name)



def predict_yield_from_img_metadata(yield_file, img_path, weather_file, out_path, 
                                    is_save_model, is_test, analyze_variety_id=2, 
                                    analyze_irrigation_id=2, key_word_list = ['Ref_filled.tif'], resname='resnet18'):
    

    suffix_list_list = [[], ['LWIR_filled.tif']]

    # suffix = ['base', 'lwir']
    # suffix_list = ['LWIR_filled.tif']
    # VI_list = ['ndvi', 'ndre', 'gndvi', 'evi']
    VI_list = ['evi']


    train_col='TRAIN_75'
    for keyword in key_word_list:

        img_list, yield_pf = select_imglist_yield(yield_file, img_path, keyword, analyze_variety_id=analyze_variety_id, analyze_irrigation_id=analyze_irrigation_id)
        
        doy=int(img_path[-20:-17])
        yield_list, train_val_dataset, test_dataset, test_indices=get_train_test_img_metadata(img_list, yield_pf, doy, weather_file, train_col=train_col)

        sample_data, _ = train_val_dataset[0]
        
        doy_name = img_path[-23:-17]
        in_channel = sample_data[0].shape[0]
        # num_epochs = 120
        num_epochs = 100
        batch_size = 32

        # Initialize an empty list to store fold-wise performance
        fold_accuracies = []
        
        # Initialize a new model for each fold
        # model = CNNRegression(in_channel)
        # model = ResNetFNN(in_channel,9,1)
        # model = ResNetFNN_V01(in_channel,9,1,resname)
        # model = ResNetFNN_V1(in_channel,9,1,resname)
        model = ResNetFNN_V2(in_channel,9,1,6, resname)
        # model = ResNetFNNTranfomer_V01(in_channel,9,1,resname)
        # model = ResNetFNNTranfomer_V2(in_channel,9,1,6, resname)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        lr=0.001
        criterion = torch.nn.MSELoss()  # Mean Squared Error loss function
        # optimizer = optim.Adam(list(conv.parameters()) + list(deconv.parameters()), lr=0.001)  # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

        # optimizer.zero_grad()
        cur_time = time.time()
        model, val_mse = train_with_cross_validation(model, train_val_dataset, batch_size, num_epochs, 
                                            optimizer, criterion, is_dual_data=True)
        training_time = time.time() - cur_time
        print('test time is  : ', training_time)
        # train_with_cross_validation(model, train_val_dataset, batch_size, num_epochs, optimizer, criterion, is_dual_data=True)

        if is_save_model:
            model_name = "path/model_" + crop_var[analyze_variety_id] + "_meta(nbands="+str(in_channel)+ ')_'+ str(doy_name) + "-"+model.__class__.__name__+'_'+resname+ "_Batch=" +str(batch_size) + "_Epoch=" +str(num_epochs) + "_lr=" +str(lr)+ "_state.pth"
            # "path/model_" + model.__class__.__name__+ doy_name +"_Batch=" +str(batch_size) + "_state.pth"
            torch.save(model.state_dict(), model_name)

        cur_time = time.time()
        test_accuracy, test_prediction = validate(model, test_dataset, criterion, batch_size = batch_size, is_return_output = True, is_dual_data=True)
        print(f'validation mse is {np.sqrt(np.mean(test_accuracy))}')
        test_time = time.time() - cur_time
        print('test time is  : ', test_time)

            
        yield_data = np.array(yield_list)
        test_irrigate_data = np.array(yield_pf[yield_pf[train_col] == 0]['Irrigation_int'])
        test_variety_data = np.array(yield_pf[yield_pf[train_col] == 0]['Variety_int'])

        test_truth = yield_data[test_indices]

        name_tag = img_path[-23:-17] + ' ' + crop_var[analyze_variety_id] + ' ' + irrigate_var[analyze_irrigation_id] + ' '
        # name_tag = 'ALL Data'
        out_name = name_tag 
        # out_name = out_name 
        out_name = out_name + keyword[:-11] 
        out_name = out_name + ''

        title = name_tag + keyword[:-11].upper() + ' CNN'


        result_df=pd.DataFrame({
            'Truth': np.array(test_truth), 
            'Prediction':np.array(test_prediction),
            'Irrigation_int':test_irrigate_data,
            'Vriaty_int':test_variety_data
        })
        csv_file_path = out_path+out_name + ' metadata.csv'
        result_df.to_csv(csv_file_path, index=False)

        plot_distinct_yields(np.array(test_truth)*0.06277, np.array(test_prediction)*0.06277, test_irrigate_data, test_variety_data, title, out_path+out_name+' metadata')


def predict_yield(yield_file, img_path, out_path, predict_model, 
                  key_word_list = ['Ref_filled.tif'], analyze_variety_id=2, 
                    analyze_irrigation_id=2 ):
    
    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # key_word_list = ['Ref_filled.tif', 'RGB_filled.tif']
    # key_word_list = ['Ref_filled.tif']
    suffix_list_list = [[], ['LWIR_filled.tif']]

    # suffix = ['base', 'lwir']
    # suffix_list = ['LWIR_filled.tif']
    # VI_list = ['ndvi', 'ndre', 'gndvi', 'evi']
    VI_list = ['evi']
    train_col='TRAIN_75'
    for keyword in key_word_list:

        img_list, yield_pf = select_imglist_yield(yield_file, img_path, keyword, analyze_variety_id=2, analyze_irrigation_id=2)

        pioneer_yield_list, train_val_dataset, test_dataset, test_indices= get_train_test_img(img_list, yield_pf, train_col=train_col)
        # yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
        doy_name = img_path[-23:-17]

        in_channel = 5
        num_epochs = 200
        batch_size = 32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        criterion = torch.nn.MSELoss()  # Mean Squared Error loss function
        
        # loaded_model = ResNetRegression_V01(in_channel, 1, resname)
        # predict_model.load_state_dict(torch.load(model_dict))
        predict_model.to(device)

        test_accuracy, test_prediction = validate(predict_model, test_dataset, criterion, batch_size = batch_size, is_return_output = True)
        print(f'validation mse is {np.sqrt(np.mean(test_accuracy))}')

        name_tag = doy_name + ' ' + crop_var[analyze_variety_id] + ' ' + irrigate_var[analyze_irrigation_id]
        out_name = name_tag 
        out_name = out_name + keyword[:-11] 
        out_name = out_name + ' Resnet '

        yield_data = np.array(pioneer_yield_list)
        test_irrigate_data = np.array(yield_pf[yield_pf[train_col] == 0]['Irrigation_int'])
        test_variety_data = np.array(yield_pf[yield_pf[train_col] == 0]['Variety_int'])

        test_truth = yield_data[test_indices]

        result_df=pd.DataFrame({
            'Truth': np.array(test_truth), 
            'Prediction':np.array(test_prediction),
            'Irrigation_int':test_irrigate_data,
            'Vriaty_int':test_variety_data
        })
        csv_file_path = out_path+out_name + '.csv'
        result_df.to_csv(csv_file_path, index=False)

        title = name_tag + ' Pioneer ' + keyword[:-11].upper() + ' CNN'
        plot_distinct_yields(np.array(test_truth), np.array(test_prediction), test_irrigate_data, test_variety_data, title, out_path+out_name)


def data_prepare_select(yield_file, img_path, out_path, keyword, selection):

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

    

if __name__== "__main__":
    yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'
    out_path = '../output/'
    img_root_path = 'D:/Corn_Yield/UAV_Data_Extracted_filled/'
    path_list = ['LIRF20220628_DOY179_extracted_filled', 'LIRF20220708_DOY189_extracted_filled', 'LIRF20220716_DOY197_extracted_filled'\
                 'LIRF20220720_DOY201_extracted_filled', 'LIRF20220916_DOY259_extracted_filled', 'LIRF20220926_DOY269_extracted_filled']
    
    for path_i in path_list:
        img_path = img_root_path+path_i

        predict_yield_from_img(yield_file, img_path, out_path, True, True)


    