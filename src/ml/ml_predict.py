from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from matplotlib.font_manager import FontProperties

import pickle

import numpy as np
import os
import matplotlib.pyplot as plt

from dataset import get_ml_image    

def prepare_train_test_data(img_list, yield_pf, metadata=None, train_col='TRAIN_75', 
                            VI_list=None, suffix_list=None, vi_only=False):
    
    yield_list = list(yield_pf['Yield_Bu_Ac'])
    train_indices = list(yield_pf[yield_pf[train_col] == 1].index)
    test_indices = list(yield_pf[yield_pf[train_col] == 0].index)

    train_img_list = [img_list[i] for i in train_indices]
    test_img_list = [img_list[i] for i in test_indices]
    
    train_images = get_ml_image(train_img_list, VI_list=VI_list, 
                                suffix_list = suffix_list, is_vi_only=vi_only)
    
    test_images = get_ml_image(test_img_list, VI_list=VI_list, 
                                suffix_list = suffix_list, is_vi_only=vi_only)
    if metadata:
        train_meta = [metadata[i] for i in train_indices]
        test_meta = [metadata[i] for i in test_indices]
        
        train_images = [train_images, train_meta]
        test_images = [test_images, test_meta]
        
        
    
    return train_images, test_images, [yield_list[i] for i in train_indices], [yield_list[i] for i in test_indices]
    
    
def ml_predict_yield(train_images, train_yields, test_images, modelname, out_name, out_path: str, is_Meta = False):
    
    
    if modelname.upper() == 'LASSO':
        model = Lasso(alpha=0.3)
    elif modelname.upper() == 'LR':
        model = LinearRegression()
    elif modelname.upper() == 'RF':
        model = RandomForestRegressor(n_estimators=150, max_depth=25, random_state=42)
    elif modelname.upper() == 'GB':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif modelname.upper() == 'SVR':
        model = LinearSVR()
    elif modelname.upper() == 'XGB':
        model = xgb.XGBRegressor(n_estimators =150, max_depth=25,objective='reg:squarederror', random_state=42)
    # xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    else:
        print('not support')
        
    if is_Meta:
        trained_model = train_model_meta(model, train_images, train_yields, out_path+out_name+modelname)
    else:
        trained_model = train_model(model, train_images, train_yields, out_path+out_name+modelname)
        
    
    # model =  trained_model.steps[-1][1]
    # if modelname.upper() == 'XGB':        
    #     # Get feature importances
    #     # importances = model.get_score(importance_type='weight')
    #     importances = model.get_booster().get_score(importance_type='weight')

    #     # Convert the dictionary of feature importances to a list of tuples
    #     importances = [(int(k[1:]), v) for k, v in importances.items()]

    #     # Sort feature importances in descending order
    #     importances = sorted(importances, key=lambda x: x[1], reverse=True)

    #     # Sort feature importances in descending order
    #     # importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    #     # Print the feature ranking
    #     print("Feature ranking:")
    #     for f, importance in enumerate(importances):
    #         print(f"{f + 1}. Feature {importance[0]} ({importance[1]})")
            
    # elif modelname.upper() == 'GB' or modelname.upper() == 'RF':
    #     # Get feature importances
    #     importances = model.feature_importances_
    #     # Sort feature importances in descending order
    #     indices = np.argsort(importances)[::-1]
    #     # Print the feature ranking
    #     print("Feature ranking:")
    #     if is_Meta:
    #         for f in range(train_images[0].shape[1]+len(train_images[1][0])):
    #             print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")
    #     else: 
    #         for f in range(train_images.shape[1]):
    #             print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")
    
    

    # Save the model to a file using pickle
    # `model_file_name` is a variable that stores the file name and path where the trained model will
    # be saved using the pickle module. The trained model is saved as a binary file with the extension
    # `.pkl`.
    model_file_name = out_path+out_name+modelname+'.pkl'
    with open(model_file_name, 'wb') as model_file:
        pickle.dump(trained_model, model_file)
        
    return trained_model

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def fit(self, X, y=None):
        #self.mean = X.mean((0,2,3)) 
        #self.std = X.std((0,2,3))
        return self

    def transform(self, X, y=None):
        # return (X-self.mean[None,:,None,None])/self.std[None,:,None,None] 
        return X

class FlattenTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # use mean value of the image 
        # X = np.nanmean(X,(2,3))
        return X.reshape((X.shape[0], -1))
    
    
class DataProcesser(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # use mean value of the image 
        # mean_image = np.nanmean(X[0],(2,3))
        # array_1 = mean_image.reshape((mean_image.shape[0], -1))
        array_1 = X[0].reshape((X[0].shape[0], -1))
        array_2 = X[1]
        return  np.concatenate((array_1, np.array(array_2)),axis=1)
        
    
def train_model_meta(model, train_images, train_yields, save_name:str = None, is_plot:bool = False):
    
    # MEAN = np.nanmean(train_images[0],(0,2,3))
    # STD = np.nanstd(train_images[0], (0,2,3))
    MEAN = 0 
    STD = 0
    print('model', model)
    
    pipe = Pipeline(steps=[("scaler", CustomScaler(MEAN,STD)),
                    ("flatten", DataProcesser()),
                    ("classifier", model )
                    ])
    pipe.fit(train_images,train_yields)

    # predict train
    pred_train = pipe.predict(train_images)
    
    if is_plot:
        plot_result(train_yields, pred_train, save_name + ' Train')
        save_result(train_yields, pred_train, save_name + ' Train')
    
    return pipe
    


def train_model(model, train_images, train_yields, save_name:str = None, is_plot:bool = False):
    
    MEAN = np.nanmean(train_images,(0,2,3))
    STD = np.nanstd(train_images, (0,2,3))

    pipe = Pipeline(steps=[("scaler", CustomScaler(MEAN,STD)),
                        ("flatten", FlattenTransformer()),
                        ("classifier", model )
                        ])
    pipe.fit(train_images,train_yields)

    # predict train
    pred_train = pipe.predict(train_images)
    
    if is_plot:
        plot_result(train_yields, pred_train, save_name + ' Train')
        save_result(train_yields, pred_train, save_name + ' Train')
    
    return pipe


def test_model(model, test_images, test_yields, test_indices, irrigate_data, save_name):
    
    # predict
    pred_validate = model.predict(test_images)
    
    # plot_result(test_yields, pred_validate, save_name )
    plot_result_separate(test_yields, pred_validate, test_indices, irrigate_data,save_name)
    save_result(test_yields, pred_validate, save_name )
    
    




def plot_result_separate(y_test, y_pred, test_indices, irrigate_data,save_name):
    
    basename = os.path.basename(save_name)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    x = np.arange(50,300)
    plt.plot(x, x, color = 'k', ls='--', alpha=0.7, linewidth=1.25)
    
    deficit_indices = np.where(irrigate_data == 1)[0]
    full_indices = np.where(irrigate_data == 0)[0]
    # plt.scatter( y_test, y_pred, label = f'r-squared = {round(r2,2)}, rmse = {round(rmse, 2)}')
    plt.scatter( y_test[np.in1d(test_indices, deficit_indices)], 
                y_pred[np.in1d(test_indices, deficit_indices)],
                marker='x', s=25, color = '#bcbd22', label = 'Deficit irrigated')
    plt.scatter( y_test[np.in1d(test_indices, full_indices)], 
                y_pred[np.in1d(test_indices, full_indices)],
                label = 'Fully irrigated', 
                marker='o', alpha=0.5, s=25, color = '#2ca02c', edgecolors='#2ca02c')
    # , facecolor='none'
    
    plt.xlim([50,300])
    plt.ylim([50,300])
    plt.xticks(fontname='Times New Roman', size=10)
    plt.yticks(fontname='Times New Roman', size=10)
    plt.xlabel('True Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)
    plt.ylabel('Predicted Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)

    plt.title(basename, fontname = 'Times New Roman', fontsize = 14)

    plt.text(55, 288, f'R-squared = {round(r2,2)}', 
             fontname = 'Times New Roman', fontsize = 11)
    plt.text(55, 275, f'RMSE = {round(rmse, 2)} (Bu/Ac)',  
             fontname = 'Times New Roman', fontsize = 11)
    
    font_props = FontProperties(family='Times New Roman', size=11)
    plt.legend(loc='lower right', frameon=False, 
               prop=font_props)
    # yy_pred = model_lasso.predict(X_train)
    # plt.scatter(y_train, yy_pred)
    plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    print(save_name, f' r-squared = {r2}, rmse = {rmse}, mae={mae}')
    
    
def plot_result(y_test, y_pred, save_name):
    
    basename = os.path.basename(save_name)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    x = np.arange(50,300)
    plt.plot(x, x, color = 'k', ls='--')
    
    plt.scatter( y_test, y_pred, label = f'r-squared = {round(r2,2)}, rmse = {round(rmse, 2)}')
    plt.xlim([50,300])
    plt.ylim([50,300])
    plt.xticks(fontname='Times New Roman', size=10)
    plt.yticks(fontname='Times New Roman', size=10)
    plt.xlabel('True Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)
    plt.ylabel('Predicted Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)


    plt.title(basename, fontname = 'Times New Roman', fontsize = 14)


    plt.text(55, 288, f'R-squared = {round(r2,2)}', fontname = 'Times New Roman', fontsize = 11)
    plt.text(55, 275, f'RMSE = {round(rmse, 2)} (Bu/Ac)',  fontname = 'Times New Roman', fontsize = 11)
    # yy_pred = model_lasso.predict(X_train)
    # plt.scatter(y_train, yy_pred)
    plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(save_name, f' r-squared = {r2}, rmse = {rmse}')
    
    
def save_result(y_test, y_pred, save_name):
    
    # Combine arrays horizontally
    # combined_array = np.hstack(( y_pred, y_test))
    combined_array = np.column_stack(( y_test, y_pred))

    # Save combined_array to a CSV file
    # np.savetxt(save_name, combined_array, delimiter=', ', fmt='%d')
    np.savetxt(save_name+'.csv', combined_array, delimiter=' ')

