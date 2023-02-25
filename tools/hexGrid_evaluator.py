import os
import sys
import codecs
import collections
import pandas as pd
import numpy as np
import jenkspy
from sklearn.neighbors import NearestNeighbors

file_name = input('Please enter a name of csv file for input >> ')
if os.path.exists(file_name):
    with codecs.open(file_name, 'r', 'utf-8', 'ignore') as f:
        df_xy = pd.read_csv(f)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
else:
    print('The processing was cancelled due to an invalid input. Please enter a correct file name')
    sys.exit()

grid_input = input('Input size of a side of grid (default=100) >> ')
try:
    grid_qt = int(grid_input)
except:
    grid_qt = 100
    print("The size of a side of grid is set to 100, due to an invalid input.")

d_input = input('Input distance decay parameter (default=3) >> ')
try:
    d_cost = int(d_input)
except:
    d_cost = 3
    print('Distance decay parameter is set to 3, due to an invalid input.')

log_doc = "Documents : " + str(len(df_xy))
print(log_doc)

min_X = df_xy["x"].min()
max_X = df_xy["x"].max()
min_Y = df_xy["y"].min()
max_Y = df_xy["y"].max()

x_coord = []
y_coord = []
x_side = ((max_X - min_X) / (grid_qt - 1))
a_side = 2/3 * x_side
y_side = (a_side * np.sqrt(3))
grid_count = 0
s_count = -1
for i in range(grid_qt+2):
    s = min_X + (x_side * (i-1))
    s_count += 1
    t = min_Y - y_side if s_count % 2 == 0 else min_Y - (y_side * 0.5)
    while t < max_Y + y_side:
        x_coord.append(s)
        y_coord.append(t)
        t += y_side
        grid_count += 1

df_grid_id = pd.DataFrame()
df_grid_id['grid_id'] = pd.RangeIndex(start=0, stop=grid_count, step=1)
print("The number of grid :" + str(grid_count+1))
df_grid_id["x"] = x_coord
df_grid_id["y"] = y_coord

count_density_all = []

for i in range(len(df_grid_id)):
    grid_x = df_grid_id["x"][i]
    grid_y = df_grid_id["y"][i]
    grid_dist = ((1 + (((df_xy["x"] - grid_x) ** 2 + (df_xy["y"] - grid_y) ** 2) ** 0.5) / a_side) ** d_cost)
    df_cnt_all = df_xy["size"]  / grid_dist
    count_density_all.append(df_cnt_all.sum())
    sys.stdout.write("\r%d/%d is done." % (i+1, len(df_grid_id)))

print("\ncreating break points of values...")
df_grid_id["density_all"] = list(map(lambda x: (x*1000).round(4), count_density_all))

num_breaks = 9
bin_breaks_all = jenkspy.jenks_breaks(df_grid_id["density_all"], nb_class=num_breaks)

df_grid_id["jenks_all"] = pd.cut(df_grid_id["density_all"], bins=bin_breaks_all, labels=False, include_lowest=False, duplicates='drop')

print("\ncreating grid-document relations...")
nearest_point = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(df_grid_id[["x", "y"]].values)
distances, indices = nearest_point.kneighbors(df_xy[["x", "y"]])
df_nearest = pd.DataFrame(indices, columns=["grid_id"])
df_xyn = pd.concat([df_xy, df_nearest], axis=1)

grid_uid = df_xyn['grid_id'].unique()
grid_doc_uid = []
grid_doc_i = []
grid_key_i = []

for i in grid_uid:
    key_i_list = []
    df_key_uid = df_xyn.query("grid_id == @i")
    keys_i = np.array(df_key_uid["keywords"])
    s = keys_i.T

    for j in s:
        try:
            l = [x.strip() for x in j.split("|")]
            key_i_list.extend(l)
        except:
            pass
    
    num_topKey = 4
    ci = collections.Counter(key_i_list)
    if len(ci) > num_topKey - 1:
        ci_top = ci.most_common(num_topKey)[0:num_topKey]
        ci_top, ci_top_counts = zip(*ci.most_common(num_topKey)[0:num_topKey])
    elif len(ci) > 0:
        ci_top = ci.most_common(len(ci))
        ci_top, ci_top_counts = zip(*ci.most_common(len(ci)))
    else:
        ci_top = "N/A"
    
    grid_doc_uid.append(i)
    grid_doc_i.append(df_key_uid['size'].sum())
    grid_key_i.append(ci_top)

df_grid_key = pd.DataFrame()
df_grid_key["grid_id"] = grid_doc_uid
df_grid_key["size"] = grid_doc_i
df_grid_key["keywords"] = grid_key_i

df_grid = pd.merge(df_grid_id, df_grid_key, on='grid_id', how='outer', sort=True)

f_name = base_name +"_grid_" + str(len(df_grid_id)) + ".tsv"
df_grid.to_csv(f_name, sep='\t', index=False, encoding='utf-8')

print("Completed.")
