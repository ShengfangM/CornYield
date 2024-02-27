from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


crop_var={
    3: 'all',
    2: 'Pioneer',
    1: 'CH 192-10',
    0: 'DKC 51-91'
}

irrigate_var={
    2: 'All',
    0: 'Full',
    1: 'Deficit'
}


def plot_distinct_yields(y_test, y_pred, irrigate_data, variety_data, title, save_name):
    
    value_range = [4,19]   
    accurate_metric = []
    
    x = np.arange(value_range[0], value_range[1]+1)
    plt.plot(x, x, color = 'k', ls='--', alpha=0.7, linewidth=1.25)
    
    color_list=['#2ca02c', '#bcbd22', 'r']
    marker_list = ['d', 'o', '+']
    # marker_list = ['o', '+','d']
    
    for j in np.sort(np.unique(variety_data)):
        marker=marker_list[j]
        for i in np.sort(np.unique(irrigate_data)):
            marker=marker_list[i+1]
            color = color_list[i]
        # marker=marker_list[i]
            # color = color_list[j]
            label = irrigate_var[i] + ' ' + crop_var[j] 

            condition = (irrigate_data == i) & (variety_data == j)

            if marker != '+':
                plt.scatter( y_test[condition], 
                        y_pred[condition],
                        marker=marker, s=25, color = color, label = label, alpha=0.5, edgecolors=color)
            else:
                plt.scatter( y_test[condition], 
                        y_pred[condition],
                        marker=marker, s=25, color = color, label = label)
            
            rmse=np.sqrt(mean_squared_error(y_test[condition], y_pred[condition]))
            mae=mean_absolute_error(y_test[condition], y_pred[condition])
            r2=r2_score(y_test[condition], y_pred[condition])

            accurate_metric.append([label, rmse, mae, r2, np.mean(y_test), 100*rmse/np.mean(y_test), 100*mae/np.mean(y_test)])

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mare = np.mean(np.abs((y_test - y_pred)/y_test))
    accurate_metric.append(['all', rmse, mae, r2, np.mean(y_test), 100*rmse/np.mean(y_test), 100*mae/np.mean(y_test)])
    
    plt.xlim(value_range)
    plt.ylim(value_range)
    plt.xticks(fontname='Times New Roman', size=10)
    plt.yticks(fontname='Times New Roman', size=10)
    plt.xlabel('Ground Measured Yield (ton/ha)', fontname = 'Times New Roman', fontsize = 12)
    plt.ylabel('Predicted Yield (ton/ha)', fontname = 'Times New Roman', fontsize = 12)

    plt.title(title, fontname = 'Times New Roman', fontsize = 14)

    plt.text(value_range[0]*1.1, value_range[1]*0.96, f'R-squared = {round(r2,2)}', 
             fontname = 'Times New Roman', fontsize = 11)
    plt.text(value_range[0]*1.1, value_range[1]*0.92, f'RMSE = {round(rmse, 2)}',  
             fontname = 'Times New Roman', fontsize = 11)
    # plt.text(4500, 17300, f'MAE = {round(mae, 2)} (Kg/Ha)',  
    #          fontname = 'Times New Roman', fontsize = 11)
    plt.text(value_range[0]*1.1, value_range[1]*0.88, f'MARE = {round(mare, 2)}',  
             fontname = 'Times New Roman', fontsize = 11)
    
    font_props = FontProperties(family='Times New Roman', size=11)
    plt.legend(loc='lower right', frameon=False, 
               prop=font_props)
    # yy_pred = model_lasso.predict(X_train)
    # plt.scatter(y_train, yy_pred)
    plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    
    df = pd.DataFrame(accurate_metric, columns=['Name', 'RMSE', 'MAE', 'R2', 'mean_truth', 'rmse/mean', 'mae/mean'])
    csv_file_path = save_name+'_metric.csv'
    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(save_name, f' r-squared = {r2}, rmse = {rmse}, mae={mae}', 100*mae/np.mean(y_test))