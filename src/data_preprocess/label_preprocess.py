
from utils import train_test_split_indices
import pandas as pd

out_label_file = 'D:/Corn_Yield/BL2022_Yld_label.csv'
yield_file = 'D:/Corn_Yield/BL2022_Yld.csv'

percentage_list=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
variety_list=['DKC 51-91', 'P9998', 'CH 192-10']
irrigation_list = ['Deficit ', 'Full']


yield_df = pd.read_csv(yield_file)
total_size = len(yield_df)

for percentage in percentage_list:
    
    train_indices_all =[]
    test_indices_all =[]

    for variety_id in variety_list:
        for irrigation_id in irrigation_list:
            temp_df = yield_df[(yield_df['Variety'] ==variety_id)& (yield_df['Irrigation'] ==irrigation_id)]
            train_indices, test_indices = train_test_split_indices(list(temp_df.index), percentage, random_seed=39)

            train_indices_all.extend(train_indices)
            test_indices_all.extend(test_indices)

    train_test_list = [1 if i in train_indices_all else 0 for i in range(total_size)]
    col_name='TRAIN_'+ str(int(100-percentage*100))
    yield_df[col_name] = train_test_list

yield_df.to_csv(out_label_file)


''' exam result'''
# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='DKC 51-91') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='DKC 51-91') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))

# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='DKC 51-91') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='DKC 51-91') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))


# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='P9998') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='P9998') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))

# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='P9998') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='P9998') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))


# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='CH 192-10') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==1) & (yield_pf['Variety'] =='CH 192-10') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))

# train_pio = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='CH 192-10') & (yield_pf['Irrigation'] =='Deficit ')]
# train_pio2 = yield_pf[(yield_pf['TRAIN_70'] ==0) & (yield_pf['Variety'] =='CH 192-10') & (yield_pf['Irrigation'] =='Full')]

# print(len(train_pio))
# print(len(train_pio2))