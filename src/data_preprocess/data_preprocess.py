import sys
sys.path.append('c:/Zhou/Ma/Projects/CornYield/src')

import numpy as np
import os
import rasterio

from skimage.transform import resize
from skimage import restoration
from path_utils import get_files_by_suffix, get_subdirectories
from io_utils import write_tiff
from rasterio.transform import from_origin



def resize_image(img, target_height, target_width):
    
    target_size = (target_height, target_width)
    
    img = img.transpose(1, 2, 0)
    
    nan_mask = np.isnan(img)
    filled_image = restoration.inpaint.inpaint_biharmonic(img, nan_mask)
    # filled_image = restoration.inpaint.inpaint_biharmonic(img, nan_mask, multichannel=False)
#     filled_image_uint8 = np.clip(filled_image * 255.0, 0, 255).astype(np.uint8)
    
    resized_img = resize(filled_image, target_size)
    resized_img = resized_img.transpose(2, 0, 1)

    return resized_img


# The `interpolate_img` function takes in an image path, new height, and new width as parameters.
# It first retrieves a list of files with the '.tif' suffix in the given image path. Then, it
# creates an output path by appending '_filled' to the input path. If the output path does not
# exist, it creates the directory.
def interpolate_images(img_path, new_height, new_width, suffix):
    
    file_list = get_files_by_suffix(img_path, suffix)
    file_list = [filename for filename in file_list if 'LWIR.tif' in filename]
    
    in_path = os.path.dirname(img_path)
    in_subpath = os.path.basename(img_path)
    out_path = os.path.join(in_path + '_filled', in_subpath + '_filled')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for (i, file) in enumerate(file_list):
        basename = os.path.basename(file)
        id_name = basename[0:12]

        src = rasterio.open(file)
        
        geotransform = src.transform
        # print(geotransform)
        new_pixel_width = geotransform[0] * src.width / new_width
        new_pixel_height = geotransform[4] * src.height / new_height
        # new_transform = from_origin(geotransform[2], geotransform[5], new_pixel_width, new_pixel_height)
        new_transform = (new_pixel_width, geotransform[1], geotransform[2],
                         geotransform[3], new_pixel_height, geotransform[5])
        
        img_array = src.read().astype(np.float32)
        img_array = img_array/100
        # 

        img_array[img_array<0] = np.nan
        resized_img = resize_image(img_array, new_height , new_width)
        
        output_file = basename[:-4]+'_filled2.tif'
        output_file = os.path.join(out_path, output_file)
        write_tiff(output_file, resized_img, src.crs, new_transform)
        
        
def interpolate_data_batch(img_path):
    sub_paths = get_subdirectories(img_path)
    for path in sub_paths:
        interpolate_images(path, 220, 55, 'tif')
        


if __name__ == "__main__":
    # img_path = 'C:/Users/yutzhou/Desktop/Corn_Yield/UAV_Data_Extracted'
    # interpolate_data_batch(img_path)
    img_path = 'C:/Users/yutzhou/Desktop/Corn_Yield/UAV_Data_Extracted\LIRF20220926_DOY269_extracted'
    interpolate_images(img_path, 220, 55, 'tif')