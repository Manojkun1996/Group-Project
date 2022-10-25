
import pandas as pd
import numpy as np
import itertools 
pd.options.mode.chained_assignment = None  # default='warn'
data1 = pd.read_csv('left_backPose.txt', sep=",", header=None)
data1 = data1[[0, 5,6,7]]
data1.columns = ["timestamp", "a_left_backPose", "b_left_backPose","c_left_backPose"]
# print(data1)

data2 = pd.read_csv('right_backPose.txt', sep=",", header=None)
data2 = data2[[0, 5,6,7]]
data2.columns = ["timestamp", "a_right_backPose", "b_right_backPose","c_right_backPose"]
# print(data2)

data3 = pd.read_csv('left_wristPose.txt', sep=",", header=None)
data3 = data3[[0, 5,6,7]]
data3.columns = ["timestamp", "a_left_wristPose", "b_left_wristPose","c_left_wristPose"]
# print(data3)

data4 = pd.read_csv('right_wristPose.txt', sep=",", header=None)
data4 = data4[[0, 5,6,7]]
data4.columns = ["timestamp", "a_right_wristPose", "b_right_wristPose","c_right_wristPose"]
print(data4)

data5 = pd.read_csv('my_data1.csv')
print(data5)

# df = pd.merge_asof(data5, data4, on='timestamp', direction='backward')
# print(df)

# from scipy.spatial import cKDTree
# def spatial_merge_NN(df1, df2, xyz=['timestamp']):
#     ''' Add features from df2 to df1, taking closest point '''
#     tree = cKDTree(df2[xyz].values)
#     dists, indices = tree.query(df1[['timestamp']].values, k=1)
#     fts = [c for c in df2.columns]
#     for c in fts:
#         df1[c] = df2[c].values[indices]
#     return df1

# df_new = spatial_merge_NN(data5, data4, ['timestamp'])
# print(df_new)
def merge_data(df1,df2):
    df3 = pd.DataFrame()
    for k, v in df1.iterrows(): 
        i = ((df2['timestamp']-v['timestamp'])).abs().idxmin() 
        df3 = df3.append(df2.loc[i])
    df3.reset_index(drop=True, inplace=True)
    return df3
df1 = merge_data(data5, data1)
df2 = merge_data(data5, data2)
df3 = merge_data(data5, data3)
df4 = merge_data(data5, data4)
df1['File Name'] = data5['File Name']
df2['File Name'] = data5['File Name']
df3['File Name'] = data5['File Name']
df4['File Name'] = data5['File Name']

df1.to_csv(r'.\my_data_left_backPose.csv', index=False)
df2.to_csv(r'.\my_data_right_backPose.csv', index=False)
df3.to_csv(r'.\my_data_left_wristPose.csv', index=False)
df4.to_csv(r'.\my_data_right_wristPose.csv', index=False)
