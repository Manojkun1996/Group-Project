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
ind = dfd[(dfd['start'] == 0) & (dfd['end'] == 0)].index
dfd.drop(ind, inplace=True)
dfd = dfd.drop([30459, 499563, 483283, 491809, 397635, 395999, 28369, 36869, 389989, 393183])
# dfd = dfd.drop([30459])
dfd = dfd.drop(['metadata'], axis=1)
dfd.rename(columns = {'values': 'Image Description'}, inplace=True)
# print(dfd)
m=list(dfd.index.values -15)
dff=df[["metadata", 'values']].iloc[m]
dff = dff.drop(['metadata'], axis=1)

dff.rename(columns = {'values': 'File Name'}, inplace = True)
# print(dff['File Name'])
# result = pd.concat([dfd,dff],axis=1, join='inner')
# dfd = dfd.insert(2, 'File Name', dff['File Name'])
dfd = dfd.join(dff['File Name'])
dfd['File Name'] = dff['File Name'].values
# start = dfd['File Name'][dfd['start']==1].tolist()
# end = dfd['File Name'][dfd['end']==1].tolist()
ff = dfd['File Name'].str.extract('(\d+)')
fs = ff[0].astype(int).tolist()
# print(ff[0].astype(int).tolist())
start_list = fs[0::2]
end_list = fs[1::2]
# ls = []
output = []
for (a, b) in itertools.zip_longest(start_list, end_list):
    ls = list(range(a, b+1))
    output.append(['frame_'+str(x)+'.jpeg' for x in ls])
    # ls.append(output)
att = ['SCREW_RIGHT', 'IDLE', 'BOLT_LEFT', 'DELIVERY_RIGHT',  'BOLT_LEFT', 'BOLT_LEFT',  'DELIVERY_BIMANUAL', 'PICKUP_RIGHT', 'BOLT_RIGHT', 'ASSEMBLY1_BIMANUAL', 'PICKUP_LEFT', 'ASSEMBLY1_BIMANUAL', 'HANDOVER_LEFT', 'SCREW_RIGHT', 'SCREW_RIGHT', 'PICKUP_LEFT-START',  'PICKUP_LEFT-START', 'ASSEMBLY2_RIGHT',  'DELIVERY_RIGHT', 'DELIVERY_RIGHT', 'SCREW_RIGHT', 'SCREW_RIGHT']

df = pd.DataFrame(output)

df = df.transpose()
df.columns = att
df = df.melt()
one_hot = pd.get_dummies(df.variable)
df = df.drop('variable', axis=1)
df = df.join(one_hot)
df = df.dropna()
df.rename(columns = {'value':'File Name'}, inplace = True)
# print(dfd['Image Description'].tolist())
# print(output)
# print(df)

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
fd.to_csv(r'G:\dataset\s01\my_data1.csv', index=False)

df3 = pd.merge(df,fd,on="File Name",how="left")
# df3["timestamp"] = df3["timestamp"].fillna("notmatched")
# shift column 'C' to first position
first_column = df3.pop('timestamp')
# insert column using insert(position,column_name,first_column) function
df3.insert(0, 'timestamp', first_column)
print(df3)
df3.to_csv(r'G:\dataset\s01\my_data2.csv', index=False)
