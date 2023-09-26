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
        corn_data = read_img(self.file_list[idx], VI_list = self.VI_list, 
                            suffix_list = self.suffix_list)
                
        if self.transform:
            corn_data = self.transform(corn_data)
                
        return torch.tensor(corn_data), torch.tensor(corn_yield)
        
        
        
        
        