import csv
import rasterio


def read_csv_to_list(csv_file, n_header : int =1, column_idx_list: list = None) -> list:
    data_list = []    
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        for _ in range(n_header):
            next(csv_reader)  # skip header
            
        for row in csv_reader:            
            row = [row[i] for i in column_idx_list] if column_idx_list else row 
            data_list.append(row)
        
    return data_list
        
        
def read_csv_pair_columns(csv_file, col1, col2):
    '''read two columns from the csv and return a dictionary
    col1 as key and col2 as value
    '''
    yield_dict = dict()
    with open(csv_file, newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            yield_dict[row[col1]] = row[col2]
    return yield_dict


def write_tiff(output_file, image_data, crs, geotransform):
#     geotransform = (top_left_x, pixel_width, 0, top_left_y, 0, -pixel_height)  # (x-coordinate of top-left corner, pixel width, rotation, y-coordinate of top-left corner, rotation, negative pixel height)

    # Create the metadata for the TIFF file
    metadata = {
        'width': image_data.shape[2],
        'height': image_data.shape[1],
        'count': image_data.shape[0],
        'dtype': image_data.dtype,
        'crs': crs,
        'transform': geotransform
    }
    
    with rasterio.open(output_file, 'w', **metadata) as dst:
    # Write the image data to the TIFF file
        for band_idx in range(image_data.shape[0]):
            dst.write(image_data[band_idx, :, :], band_idx + 1) 
        # dst.write(image_data, indexes=list(range(1, num_bands + 1)))