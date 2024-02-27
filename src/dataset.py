import numpy as np
import pandas as pd
import rasterio
import os
from io_utils import read_csv_to_list
from path_utils import get_files_with_matching_word
from vegetation_indices import calculate_evi, calculate_gndvi, calculate_ndre,calculate_ndvi

from config_constant import IRRIGATE_IDS, CROP_IDS, CROP_TYPE, \
                            id_idx, variety_idx, irrigate_idx, yield_idx



def get_imgfilelist_yield(data_path:str, yield_file:str, key_word:str= 'Ref_filled.tif', 
                               crop_type_select:list= None, irrigate_type_select: list = None):
    
    data_files = get_files_with_matching_word(data_path, key_word)
    data_id_list = [os.path.basename(file_name)[:12] for file_name in data_files]
    sort_id_index = dict(zip(data_id_list, range(len(data_id_list))))

    yield_pf = pd.read_csv(yield_file)
    yield_pf = yield_pf.sort_values(by=['ID'], key=lambda x: x.map(sort_id_index))
    yield_pf['Irrigation_int'] = yield_pf['Irrigation'].map(IRRIGATE_IDS)
    yield_pf['Variety_int'] = yield_pf['Variety'].map(CROP_IDS)

    if crop_type_select:
        yield_pf = yield_pf[yield_pf['Variety'].isin(crop_type_select)]
        indices = yield_pf.index.tolist()
        data_files = [data_files[i] for i in indices]

    if irrigate_type_select:
        yield_pf = yield_pf[yield_pf['Irrigation'].isin(irrigate_type_select)]
        indices = yield_pf.index.tolist()
        data_files = [data_files[i] for i in indices]

    return data_files, yield_pf


def create_metadata(yield_pf, weather_file, doy):

    weather_pf = pd.read_excel(weather_file,sheet_name=1, header=[0,1,3])
    # header_name = weather_pf.columns
    # selected_rows = weather_pf[(weather_pf[header_name[-1]] > doy-6) & (weather_pf[header_name[-1]] < doy+6)]
    weather_pf = pd.read_excel(weather_file,sheet_name=1, header=[0,1,3])
    weather_pf = weather_pf.dropna()

    selected_rows = weather_pf[(weather_pf.iloc[:,-1] > doy-6) & (weather_pf.iloc[:,-1]  < doy+6) ]

    air_temperature = selected_rows.iloc[:,2].mean()
    precipitation = selected_rows.iloc[:,9].mean()
    soil_temperature = selected_rows.iloc[:,10].mean()
    soil_temperature2 = selected_rows.iloc[:,11].mean()

    metadata = pd.DataFrame(index=range(len(yield_pf)))
    metadata['Variety'] = yield_pf['Variety_int']/2
    metadata['Irrigation'] = yield_pf['Irrigation_int']/1
    metadata['DOY'] = doy / 366
    metadata['Month'] = (doy / 30)/12
    metadata['Stage'] = (doy/30 - 5)/5
    metadata['air_temperature'] = air_temperature
    metadata['precipitation'] = precipitation
    metadata['soil_temperature'] = soil_temperature
    metadata['soil_temperature2'] = soil_temperature2
    return metadata


def read_img(img_file, VI_list = None, suffix_list = None, is_vi_only:bool = False):
    
    with rasterio.open(img_file) as src:
        src_data = src.read()
        
    if is_vi_only:
        all_data = None
    else:
        all_data = src_data
        
    if VI_list:
        for vi in VI_list:
            if vi == 'ndvi':
                ndvi = calculate_ndvi(src_data[4,:,:] , src_data[2,:,:])
                ndvi = (ndvi+1.0)/2.0
                vi_data = ndvi[np.newaxis,:,:]
                
            elif vi == 'ndre':
                ndre = calculate_ndre(src_data[4,:,:], src_data[3,:,:] )
                ndre = (ndre+1.0)/2.0
                vi_data = ndre[np.newaxis,:,:]
                
            elif vi == 'gndvi':
                gndvi = calculate_gndvi(src_data[4,:,:], src_data[1,:,:] )
                gndvi = (gndvi+1.0)/2.0
                vi_data = gndvi[np.newaxis,:,:]
                
            elif vi == 'evi':
                evi = calculate_evi(src_data[4,:,:], src_data[2,:,:], src_data[0,:,:] )
                evi = (evi+1.0)/2.0
                vi_data = evi[np.newaxis,:,:]
                
            try:    
                all_data = np.append(all_data, vi_data, axis=0)
            except:
                all_data = vi_data
    del src_data 
    basename = os.path.basename(img_file)[:33]
    basename = os.path.join(os.path.dirname(img_file), basename)
    if suffix_list:
        for suffix in suffix_list:
            with rasterio.open(basename + suffix) as src:
                all_data = np.append(all_data, src.read(), axis=0)
                # all_data = np.vstack( all_data, src.read(),axis=0)
               
    return all_data


def get_ml_image(img_list, VI_list = None, suffix_list = None, 
                 is_vi_only:bool = False) -> np.array:
    
    all_image = []
    for img_file in img_list:
        img = read_img(img_file, VI_list = VI_list, 
                       suffix_list = suffix_list, is_vi_only=is_vi_only)    
        all_image.append(img)
    
    return np.array(all_image)



def get_ordered_yields_from_filelist(yield_dict, data_file_list, yield_idx = 6):
    
    ordered_yields = []
    irrigate_type = []
    for i, filepath in enumerate(data_file_list):
        file_name = os.path.basename(filepath)
        file_id = file_name[:12]
        ordered_yields.append(yield_dict[file_id][0])
        irrigate_type.append(1) if yield_dict[file_id][2] == 'Deficit ' else irrigate_type.append(0)

    return ordered_yields, irrigate_type


def select_data_and_yield_list(data_path:str, yield_file:str, key_word:str= 'Ref_filled.tif', 
                               crop_type_select:list= None, irrigate_type_select: list = None):
  
    yield_list = read_csv_to_list(yield_file)
    yield_dict = {data[id_idx]:[float(data[yield_idx]), data[variety_idx], 
                                data[irrigate_idx]] for data in yield_list}
    
    data_files = get_files_with_matching_word(data_path, key_word)
    crop_files = []
    crop_yields = []
    irrigate_type = []    
    
    if crop_type_select:
        for data_file in data_files:
            file_name = os.path.basename(data_file)
            file_id = file_name[:12]
            for crop_type in crop_type_select:
                if yield_dict[file_id][1] == CROP_TYPE[crop_type.lower()]:
                    crop_files.append(data_file)
                    crop_yields.append(yield_dict[file_id][0])
                    irrigate_type.append(1) if yield_dict[file_id][2] == 'Deficit ' else irrigate_type.append(0)
                    break
        
    else:
        crop_files = data_files
        crop_yields, irrigate_type = get_ordered_yields_from_filelist(yield_dict, data_files)
        
    # for irrigate in irrigate_type_select:
    #     irrigate_indices = np.where(IRRIGATE_IDS[irrigate])
        
    return crop_files, crop_yields, irrigate_type
    
