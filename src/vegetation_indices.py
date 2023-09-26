import numpy as np

def calculate_ndvi(nir_band: np.array, red_band: np.array) -> np.array:
    ndvi = (nir_band - red_band)/(nir_band + red_band)
    ndvi[ndvi == np.nan] = -1
    return ndvi

def calculate_ndre(nir_band: np.array, rededge_band: np.array) -> np.array:
    ndre = (nir_band - rededge_band)/(nir_band + rededge_band)
    ndre[ndre == np.nan] = -1
    return ndre


def calculate_gndvi(nir_band: np.array, green_band: np.array) -> np.array:
    gndvi = (nir_band - green_band)/(nir_band + green_band)
    gndvi[gndvi == np.nan] = -1
    return gndvi

def calculate_evi(nir_band: np.array, red_band: np.array, blue_band: np.array) -> np.array:
    ''' the value range of band must be 0-1'''
    evi = 2.5*((nir_band - red_band)/(nir_band + 6*red_band-7.5*blue_band+1) )
    return evi