"""
@author: Manoj
"""

import pandas as pd
import numpy as np
import itertools
import ntpath
pd.options.mode.chained_assignment = None  # default='warn'

# Metadata of all the images is extracted as a csv file using the metadata extractor.

#Enter the location of the csv file which contains the extracted metadata
FILE_LOCATION = r"G:\dataset\s01\out.csv" #out.csv is the file containing extracted metadata from the images
df = pd.read_csv(FILE_LOCATION, error_bad_lines=False, names=['metadata', 'junk', 'value'])

df = df.drop('junk', axis=1)

# df['metadata'] = df['metada1ta'].astype(str)
df = df['metadata'].str.split(':', n=1, expand=True)

df.columns = ['metadata', 'values']

df['start'] = df["values"].str.contains("START", case=False, na=False).astype(int)

# df['start'] = df['values'].apply(lambda x: 'START' in x.lower())
df['end'] = df["values"].str.contains("END", case=False, na=False).astype(int)

dfd = df[df['metadata'].str.contains('Image Description')]
# dfd['values']=dfd['values'].replace('  ', np.nan, inplace=True)
# dfd= dfd.dropna(subset=['values'], inplace=True)
# print(dfd.shape)
ind = dfd[(dfd['start'] == 0) & (dfd['end'] == 0)].index
dfd.drop(ind, inplace=True)
# dfd = dfd.drop([30459, 499563, 483283, 491809, 397635, 395999, 28369, 36869, 389989, 393183])
# dfd = dfd.drop([30459])
dfd = dfd.drop(['metadata'], axis=1)
# print(dfd.shape)
dfd.rename(columns = {'values': 'Image Description'}, inplace=True)

m=list(dfd.index.values -15)
# print(m)
dff=df[["metadata", 'values']].iloc[m]
dff = dff.drop(['metadata'], axis=1)
# print(dff)
dff.rename(columns = {'values': 'File Name'}, inplace = True)
# print(dff['File Name'])
# result = pd.concat([dfd,dff],axis=1, join='inner')
# dfd = dfd.insert(2, 'File Name', dff['File Name'])
dfd = dfd.join(dff['File Name'])


dfd['File Name'] = dff['File Name'].values
# print(dfd)
# start = dfd['File Name'][dfd['start']==1].tolist()
# end = dfd['File Name'][dfd['end']==1].tolist()
ff = dfd['File Name'].str.extract('(\d+)')

ff.columns = ['filenumber']
ff = ff.astype({"filenumber": int})
# print(ff)
ff = ff.sort_values(by='filenumber')
# print(ff)

ff3 =pd.concat([dfd, ff], axis=1)
# print(ff3)
ff3 = ff3.sort_values('filenumber')
# print(ff3)

fs = ff3['filenumber'].astype(int).tolist()
# print(ff3['filenumber'].astype(int).tolist())
start_list = fs[0::2]
end_list = fs[1::2]
#print(start_list)
#print(end_list)
subtracted = [element1 - element2 for (element1, element2) in zip(end_list, start_list)]
#print(sum(subtracted))

# ls = []

output = [] # list of lists of image names in between start and stop
for (a, b) in itertools.zip_longest(start_list, end_list):
    ls = list(range(a, b+1))
    output.append(['frame_'+str(x)+'.jpeg' for x in ls])
    # ls.append(output)
# print(len(output))
# att = ['SCREW_RIGHT', 'IDLE', 'BOLT_LEFT', 'DELIVERY_RIGHT',  'BOLT_LEFT', 'BOLT_LEFT',  'DELIVERY_BIMANUAL', 'PICKUP_RIGHT', 'BOLT_RIGHT', 'ASSEMBLY1_BIMANUAL', 'PICKUP_LEFT', 'ASSEMBLY1_BIMANUAL', 'HANDOVER_LEFT', 'SCREW_RIGHT', 'SCREW_RIGHT', 'PICKUP_LEFT-START',  'PICKUP_LEFT-START', 'ASSEMBLY2_RIGHT',  'DELIVERY_RIGHT', 'DELIVERY_RIGHT', 'SCREW_RIGHT', 'SCREW_RIGHT']
# print(len(att))

lst=ff3['Image Description'].tolist()
#print(lst)
att = []
for s in lst:
    att.append(s.replace('-START', '').replace('-END', ''))

att = att[0::2]
#print(list(set(att)))

df = pd.DataFrame(output)
# print(df)
df = df.transpose()
#print(df)
df.columns = att
df = df.melt()
#print(df)
one_hot = pd.get_dummies(df.variable)
df = df.drop('variable', axis=1)
#print(df)
df = df.join(one_hot)
df = df.dropna()
df.rename(columns = {'value':'File Name'}, inplace = True)

# print(output)
#print(df)
""" New data frame """
data = pd.read_csv('frames_log.txt', sep=",", header=None)
data.columns = ["timestamp", "File Address"]


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
# data['File Name'] = path_leaf(data['File Address'])
fname = []
for ind in data.index:
    fname.append(path_leaf(data['File Address'][ind]))
data['File Name'] = fname
data = data.drop( "File Address",axis=1)
# print(df)

dd = {k: g["timestamp"].tolist()[0] for k,g in data.groupby("File Name")}
# print(len(dd.keys()))
# print(data)
# USERS = pd.merge(df, data, on=["File Name"], how="outer", indicator=True)
# USERS = USERS.loc[USERS["_merge"] == "left_only"].drop("_merge", axis=1)
# print(USERS)


dict_filter = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

cc = df['File Name'].tolist()
# ff = []
# for i in cc:
#     ff.append()dd[i]

# print(ff)
# new_dict_keys = ("c","d")
small_dict=dict_filter(dd, cc)
# print(len(small_dict.keys()))
fd= pd.DataFrame(small_dict.items(), columns=['File Name', 'timestamp'])
# fd["timestamp"] = fd["timestamp"].explode().astype(int)
print(fd)
# fd.to_csv(r'G:\dataset\s02_nh\my_data1.csv', index=False)

df3 = pd.merge(df,fd,on="File Name",how="left")
# df3["timestamp"] = df3["timestamp"].fillna("notmatched")
# shift column 'C' to first position
first_column = df3.pop('timestamp')
# insert column using insert(position,column_name,first_column) function
df3.insert(0, 'timestamp', first_column)
#print(df3.columns)
#df3.to_csv(r'G:\dataset\s02_nh\my_data2.csv', index=False)


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
#print(data4)

data5 = fd
#print(data5)
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

print(df1)
print(df2)
print(df3)
print(df4)

df1.to_csv(r'.\my_data_left_backPose.csv', index=False)
df2.to_csv(r'.\my_data_right_backPose.csv', index=False)
df3.to_csv(r'.\my_data_left_wristPose.csv', index=False)
df4.to_csv(r'.\my_data_right_wristPose.csv', index=False)

