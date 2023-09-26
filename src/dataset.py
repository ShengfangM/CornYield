import numpy as np
import rasterio
import os
from io_utils import read_csv_to_list
from src.path_utils import get_files_with_matching_word
from src.vegetation_indices import calculate_evi, calculate_gndvi, calculate_ndre,calculate_ndvi

id_idx = 1
variety_idx = 3
irrigate_idx = 4
yield_idx = 6
CROP_TYPE = {
    'pioneer':'P9998'
}
IRRIGATE_IDS = {
    'full':0,
    'deficit':1
}

def get_ordered_yields_from_filelist(yield_dict, data_file_list, yield_idx = 6):
    
    ordered_yields = []
    irrigate_type = []
    for i, filepath in enumerate(data_file_list):
        file_name = os.path.basename(filepath)
        file_id = file_name[:12]
        ordered_yields.append(yield_dict[file_id][0])
        irrigate_type.append(1) if yield_dict[file_id][2] == 'Deficit ' else irrigate_type.append(0)

    return ordered_yields, irrigate_type


def select_data_and_yield_list(data_path: str, yield_file: str, key_word : str= 'Ref_filled.tif', 
                               crop_type_select : list= None, irrigate_type_select: list = None):
  
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
                vi_data = ndvi[np.newaxis,:,:]
                
            elif vi == 'ndre':
                ndre = calculate_ndre(src_data[4,:,:], src_data[3,:,:] )
                vi_data = ndre[np.newaxis,:,:]
                
            elif vi == 'gndvi':
                gndvi = calculate_gndvi(src_data[4,:,:], src_data[1,:,:] )
                vi_data = gndvi[np.newaxis,:,:]
                
            elif vi == 'evi':
                evi = calculate_evi(src_data[4,:,:], src_data[2,:,:], src_data[0,:,:] )
                vi_data = evi[np.newaxis,:,:]
                
            try:    
                all_data = np.append(all_data, vi_data, axis=0)
            except:
                all_data = vi_data

    basename = img_file[:-14]
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



# def get_data_pioneer_indexed(file_paths, yield_dict, pioneer_deficit_id, pioneer_full_id):
#     all_dataset = []
#     all_yield = []
#     pioneer_deficit_idx_list = []
#     pioneer_full_idx_list = []
    
#     for i, filepath in enumerate(file_paths):
#         file_name = os.path.basename(filepath)
#         file_id = file_name[:12]
#         src = rasterio.open(filepath)
#         array = src.read()
#         all_dataset.append(array)
#         all_yield.append(yield_dict[file_id])
        
#         if file_id in pioneer_deficit_id:
#             pioneer_deficit_idx_list.append(i)
        
#         elif file_id in pioneer_full_id:
#             pioneer_full_idx_list.append(i)
            
#     return np.array(all_dataset), np.array(all_yield), np.array(pioneer_deficit_idx_list), np.array(pioneer_full_idx_list)


# def select_data_and_yield_list(data_path, yield_file, key_word= 'Ref_filled.tif', 
#                                selection = None):
    
#     yield_list = read_csv_to_list(yield_file)
#     yield_dict = {data[id_idx]:[float(data[yield_idx]), data[variety_idx], data[irrigate_idx]] for data in yield_list}
    
#     data_files = get_files_with_matching_word(data_path, key_word)
    
#     if selection:
#         select_files = []
#         select_yields = []
#         select_type = []
#         if selection.lower() == 'pioneer':
#             for data_file in data_files:
#                 file_name = os.path.basename(data_file)
#                 file_id = file_name[:12]
#                 if yield_dict[file_id][1] == 'P9998':
#                     select_files.append(data_file)
#                     select_yields.append(yield_dict[file_id][0])
#                     select_type.append(1) if yield_dict[file_id][2] == 'Deficit ' else select_type.append(0) 
#         if selection.lower() == 'pioneer deficit':
#             for data_file in data_files:
#                 file_name = os.path.basename(data_file)
#                 file_id = file_name[:12]
#                 if yield_dict[file_id][1] == 'P9998' and yield_dict[file_id][2] == 'Deficit ':
#                     select_files.append(data_file)
#                     select_yields.append(yield_dict[file_id][0])   
#                     select_type.append(1)        
#         if selection.lower() == 'pioneer full':
#             for data_file in data_files:
#                 file_name = os.path.basename(data_file)
#                 file_id = file_name[:12]
                
#                 if yield_dict[file_id][1] == 'P9998' and yield_dict[file_id][2] == 'Full':
#                     select_files.append(data_file)
#                     select_yields.append(yield_dict[file_id][0])
#                     select_type.append(0)                
        
#         # select_files = []
#         # select_yields = []

#         # for data_file in data_files:
#         #     file_name = os.path.basename(data_file)
#         #     file_id = file_name[:12]
#         #     yield_type = yield_dict[file_id][1]

#         #     if selection == 'pioneer all':
#         #         if yield_type == 'P9998':
#         #             select_files.append(data_file)
#         #             select_yields.append(yield_type)
#         #     elif selection == 'pioneer deficit':
#         #         if yield_type == 'P9998' and yield_dict[file_id][2] == 'Deficit ':
#         #             select_files.append(data_file)
#         #             select_yields.append(yield_type)
#         #     elif selection == 'pioneer full':
#         #         if yield_type == 'P9998' and yield_dict[file_id][2] == 'Full':
#         #             select_files.append(data_file)
#         #             select_yields.append(yield_type)
#         return select_files, select_yields, select_type
#     else:
#         ordered_yield_list = get_ordered_yields_from_filelist(yield_list, data_files)
#         return data_files, ordered_yield_list


