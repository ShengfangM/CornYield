import torch
import rasterio
import numpy as np
from dataset import read_img
# CORN_FILE_SUFFIX = ['DEM_filled.tif', 
#                     'LWIR_filled.tif',
#                     'Ref_filled.tif',
#                     'RGB_filled.tif']
# CORN_FILE_SUFFIX = ['LWIR_filled.tif', 'DEM_filled.tif']
BASE_NAME_IDX = -14
BASE_TIME = '20220610_DOY161'
TIME_LIST = ['20220616_DOY167', '20220628_DOY179', '20220705_DOY186', '20220708_DOY189']


class CornDataset(torch.utils.data.Dataset):
    '''
    custom function to transfer corn data into torch dataset for deep learning model
    '''
    def __init__(self, ref_file_list : list, yield_list: list, 
                VI_list = None, suffix_list = None, transform = None):
        self.file_list = ref_file_list
        self.yield_list = yield_list
        self.transform = transform
        self.VI_list = VI_list
        self.suffix_list = suffix_list
        
    def __len__(self):
        return len(self.yield_list)
    
    def __getitem__(self, idx):
        
        corn_yield = self.yield_list[idx]
        corn_data = torch.from_numpy(read_img(self.file_list[idx], VI_list = self.VI_list, 
                            suffix_list = self.suffix_list))
        
        # corn_data = corn_data[2:,:,:]
                
        if self.transform:
            corn_data = self.transform(corn_data)
                
        return corn_data, torch.tensor(corn_yield)
        
    

class MixedDataset(torch.utils.data.Dataset):
    '''
    custom function to transfer corn data into torch dataset for deep learning model
    '''
    def __init__(self, ref_file_list : list, yield_list: list, meta_data_df,
                VI_list = None, suffix_list = None, transform = None):
        self.file_list = ref_file_list
        self.yield_list = yield_list
        self.metadata = meta_data_df
        self.transform = transform
        self.VI_list = VI_list
        self.suffix_list = suffix_list
    
        
    def __len__(self):
        return len(self.yield_list)
    
    def __getitem__(self, idx):
        
        corn_yield = self.yield_list[idx]
        corn_data = torch.from_numpy(read_img(self.file_list[idx], VI_list = self.VI_list, 
                            suffix_list = self.suffix_list))
        meta_data = torch.tensor(self.metadata.iloc[idx]).float()
        if self.transform:
            corn_data = self.transform(corn_data)
                
        return (corn_data, meta_data), torch.tensor(corn_yield)
        
 
class CornDatasetTimeSeries(torch.utils.data.Dataset):
    '''
    custom function to transfer corn data into torch dataset for deep learning model
    '''
    def __init__(self, ref_file_list : list, yield_list: list, 
                VI_list = None, suffix_list = None, transform = None):
        self.file_list = ref_file_list
        self.yield_list = yield_list
        self.transform = transform
        self.VI_list = VI_list
        self.suffix_list = suffix_list
        
    def __len__(self):
        return len(self.yield_list)
    
    def __getitem__(self, idx):
        
        corn_yield = self.yield_list[idx]
        current_file_name = self.file_list[idx]
        current_time = current_file_name[-30:-15]
        base_file = current_file_name.replace(current_time, BASE_TIME)
        corn_data = read_img(base_file, VI_list = self.VI_list, 
                            suffix_list = self.suffix_list)
        corn_data = np.expand_dims(corn_data, axis=0)

        for file_time in TIME_LIST:
            temp_file = base_file.replace(BASE_TIME, file_time)
            temp_data = read_img(temp_file, VI_list = self.VI_list, 
                            suffix_list = self.suffix_list)
            # corn_data = np.stack((corn_data, temp_data), axis=0)
            corn_data = np.concatenate([np.expand_dims(temp_data, axis=0), corn_data], axis=0)

            # print(corn_data.shape)

        corn_data = torch.from_numpy(corn_data)        
        if self.transform:
            corn_data = self.transform(corn_data)
                
        return corn_data, torch.tensor(corn_yield)
               
        
        