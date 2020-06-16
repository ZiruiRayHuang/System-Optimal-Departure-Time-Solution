import pandas as pd
import numpy as np
import math
from scipy import stats
import random
import os
import pickle
import sys
import copy
from tqdm import tqdm
import datetime
import time as tm


#------------------------------------ DEFINE FUNCTIONS -----------------------------------
def round_to_30(x, base=30):
    return base * round(x/base)

def info_for_original_id_to_new_id(nodes_mapping):
    mapping = []
    diff = nodes_mapping['OriginalID'].values[0] - nodes_mapping['NewID'].values[0]
    for i in range(1, len(nodes_mapping)-1):
        if (nodes_mapping['OriginalID'].values[i] - nodes_mapping['NewID'].values[i]) != diff:
            mapping.append([nodes_mapping['OriginalID'].values[i-1], diff])
            diff = nodes_mapping['OriginalID'].values[i] - nodes_mapping['NewID'].values[i]
    mapping.append([nodes_mapping['OriginalID'].values[-1], diff])
    return mapping

def original_id_to_new_id(original_id, mapping):
    #if original_id <= 800:
    #    return original_id - 294
    #elif original_id <= 5283:
    #    return original_id - 295
    #elif original_id <= 5640:
    #    return original_id - 296
    #elif original_id <= 5655:
    #    return original_id - 297
    #else:
    #    return original_id - 298
    for item in mapping:
        if original_id <= item[0]:
            return original_id - item[1]

#------------------------------------- PREPROCESSING -------------------------------------
print()
print()
print(r'        ____                                  _____   ')
print(r'       |  _ \   _   _   _ __    _   _   ___  |_   _|  ')
print(r"       | | | | | | | | | '_ \  | | | | / __|   | |    ")
print(r'       | |_| | | |_| | | | | | | |_| | \__ \   | |    ')
print(r'       |____/   \__, | |_| |_|  \__,_| |___/   |_|    ')
print(r'                |___/                                 ')
print()
print()
print('DynusT Program for System Optimal Individualized Incentive Scheme')
print('                       Version: 2020                             ')
print('                   Release Date: May, 2020                       ')
print()
print('                       Contributors:                             ')
print('             Yi-Chang Chiu, Zirui (Raymond) Huang                ')
print()
print('                      Copyrighted by:                           ')
print('                  Arizona Board of Regents                       ')
print()
print('                   Authorized to Release by:                     ')
print('                       Metropia Inc.                           ')
print()
print()
print()
print()

print('******************** PART I: PREPROCESSING *********************')
print()

pardir = os.path.abspath(os.path.join(os.path.realpath(sys.executable),os.path.pardir,os.path.pardir))

if os.path.exists(pardir + r'/Preprocessed/'):
    pass
else:
    os.mkdir(pardir + r'/Preprocessed/')
#---------------------------------- Configuration Data -----------------------------------
print('      Configuration Data      <step 1/5>          Processing... ')
path = pardir + r'/Input/config.dat'
for i in tqdm(range(1),desc='<1/1>'):
    pass
with open(path, 'r') as file:
    data = file.readlines()
start_time = int(data[0].rstrip().split()[0])
end_time = int(data[0].rstrip().split()[1])
peak = [[] for _ in range(len(data) - 3)]
for i in range(len(data) - 3):
    peak[i] = [int(data[1 + i].rstrip().split()[0]), int(data[1 + i].rstrip().split()[1])]
participate_rate = float(data[-2].rstrip()) / 100
max_permissive_hour = int(data[-1].rstrip())
print('Saving Files ...  ', end='', flush = True)
print('Done')
print()

#------------------------------------- Network Data --------------------------------------
print('        Network Data          <step 2/5>          Processing... ')
path = pardir + r'/Input/Network.dat'
path = r'/Users/huangzirui/Downloads/Individualized_Incentives/Package/Example_Datasets/M4_0429_v8/Network.dat'
with open(path, 'r') as file:
    data = file.readlines()
basic_data = [int(number) for number in data[0].split()]     #[the num of zones, the num of nodes, the num of links, the num of K-Shortest Path, User Super Zones]
link_data = [[int(data[index].split()[0]),
              int(data[index].split()[1]),
              int(data[index].split()[4]),
              int(data[index].split()[5]),
              int(data[index].split()[8])] for index in tqdm(range(1 + basic_data[1],1 + basic_data[1] + basic_data[2]),desc='<1/2>')]
link_data = pd.DataFrame(link_data)
link_data.columns = ['FromID','ToID','Length','#Lanes','SpeedLimit']
# Change Node ID range
all_nodes_original = list(set([int(i) for i in link_data['FromID']] + [int(i) for i in link_data['ToID']]))
all_nodes_new = [(i+1) for i in range(0,len(all_nodes_original))]
nodes_mapping = pd.DataFrame({'OriginalID':all_nodes_original,'NewID':all_nodes_new})
for index in tqdm(range(0,len(nodes_mapping)),desc="<2/2>"):
    link_data.loc[link_data['FromID'] == nodes_mapping.iloc[index]['OriginalID'],'FromID'] = nodes_mapping.iloc[index]['NewID']
    link_data.loc[link_data['ToID'] == nodes_mapping.iloc[index]['OriginalID'],'ToID'] = nodes_mapping.iloc[index]['NewID']
print('Saving Files ...  ', end='',flush=True)
link_data.to_csv(pardir + r'/Preprocessed/link_data.csv')
link_data.to_csv(r'/Users/huangzirui/Downloads/Individualized_Incentives/Package/Preprocessed/link_data.csv')
nodes_mapping.to_csv(pardir + r'/Preprocessed/nodes_mapping.csv')
nodes_mapping.to_csv(r'/Users/huangzirui/Downloads/Individualized_Incentives/Package/Preprocessed/nodes_mapping.csv')
print('Done')
print()


#------------------------------------- Vehicle Data --------------------------------------
print('        Vehicle Data          <step 3/5>          Processing... ')
#--------- import_vehicle() ----------
path = pardir + r'/Input/vehicle.dat'
path = r'/Users/huangzirui/Downloads/Individualized_Incentives/Package/Example_Datasets/M4_0429_v8/output_vehicle.dat'
with open(path, 'r') as file:
    data = file.readlines()

vehicle_data = [[int(data[index].split()[0]),
                 float(data[index].split()[3]),
                 int((float(data[index].split()[3]) - 0.1) / 30)] for index in tqdm(range(2, len(data),2),desc='<1/4>')]
vehicle_data = pd.DataFrame(vehicle_data)
vehicle_data.columns = ['VehID','Stime','Stime_30min']
# Plot Departure Time Distribution
#sns.set(style="darkgrid")
#g = sns.distplot(vehicle_data['Stime'], hist=True, kde = False)
#g.set_xticks(list(range(0, 2161, 120)))
#xlabels = [int((720 + x) / 60) % 24 for x in g.get_xticks()]
#xlabels = [('{:}' + ':00').format(x) for x in xlabels]
#g.set_xticklabels(xlabels)
#g.axvspan(0, 720, ymin=0, ymax=30000, alpha=0.2, color='red')
#plt.text(50, 230000, 'Day 1', fontsize=18)
#plt.text(750, 230000, 'Day 2', fontsize=18)
#g.axvspan(720, 2160, ymin=0, ymax=30000, alpha=0.2, color='green')
#plt.title('Departure Time Distribution')
#plt.xlabel('Departure Time')
#plt.ylabel('# of Travelers')

vehicle_data = vehicle_data.loc[(vehicle_data['Stime'] > float(start_time)) & (vehicle_data['Stime'] <= float(end_time))]
with open(pardir + r'/Preprocessed/vehicle_data', 'wb') as filehandle:
    pickle.dump(vehicle_data, filehandle)
with open(pardir + r'/Users/huangzirui/Downloads/Individualized_Incentives/Package/Preprocessed/vehicle_data', 'wb') as filehandle:
    pickle.dump(vehicle_data, filehandle)
peak_list = []

for item in peak:
    peak_list = peak_list + list(vehicle_data.index[(vehicle_data['Stime_30min'] >= int(item[0] / 30)) & (vehicle_data['Stime_30min'] < int(item[1] / 30))])

average_time_window = int((max_permissive_hour / 2 * 60 - 30) / 10)
random.seed(9)
list_random = np.random.choice(peak_list, math.ceil(len(peak_list) * participate_rate), replace=False)
participates = [0] * len(vehicle_data)
tolerance_time = [0] *  len(vehicle_data)
for i in tqdm(range(len(list_random)),desc='<2/4>'):
    participates[list_random[i]] = 1
    tolerance_time[list_random[i]] = int(round_to_30(30 + stats.erlang.rvs(a=average_time_window, scale = 10)) / 30)

# Visualizing Erlang Distribution
#fig, ax = plt.subplots(1, 1)
#r = stats.erlang.rvs(a=average_time_window, scale = 10, size = 10000)
#ax.hist(r)

vehicle_data['Participate'] = participates
vehicle_data['Tolerance'] = tolerance_time

vehicle_data_Stime = vehicle_data['Stime'].tolist()
with open(pardir + r'/Preprocessed/vehicle_data_Stime', 'wb') as filehandle:
    pickle.dump(vehicle_data_Stime, filehandle)

vehicle_data_Stime_30min = vehicle_data['Stime_30min'].tolist()
with open(pardir + r'/Preprocessed/vehicle_data_Stime_30min', 'wb') as filehandle:
    pickle.dump(vehicle_data_Stime_30min, filehandle)

vehicle_data_participate = vehicle_data['Participate'].tolist()
with open(pardir + r'/Preprocessed/vehicle_data_participate', 'wb') as filehandle:
    pickle.dump(vehicle_data_participate, filehandle)

vehicle_data_tolerance = vehicle_data['Tolerance'].tolist()
with open(pardir + r'/Preprocessed/vehicle_data_tolerance', 'wb') as filehandle:
    pickle.dump(vehicle_data_tolerance, filehandle)


#--------- import_path() ----------

path = pardir + r'/Input/path.dat'
with open(path, 'r') as file:
    data = file.readlines()

path_data = [[]] * len(vehicle_data)

for index in tqdm(peak_list,desc='<3/4>'):
    path_data[index] = data[index].split()

mapping = info_for_original_id_to_new_id(nodes_mapping)

with open(pardir + r'/Preprocessed/mapping.dat','wb') as filehandle:
# store the data as binary data stream
    pickle.dump(mapping, filehandle)

for index in tqdm(peak_list,desc='<4/4>'):
    path_data[index] = [original_id_to_new_id(int(path_data[index][j]), mapping) for j in range(len(path_data[index]))]

print('Saving Files ...  ', end='', flush = True)
with open(pardir + r'/Preprocessed/path_data.dat', 'wb') as filehandle:
        # store the data as binary data stream
    pickle.dump(path_data, filehandle)
print('Done')
print()

#------------------------------------- Volume Data --------------------------------------
print('        Volume Data          <step 4/5>          Processing... ')
#---------- import_vol() -----------
num_links = len(link_data)
path = pardir + r'/Input/OutAccuVol.dat'
with open(path, 'r') as file:
    data = file.readlines()
num_time_intervals = math.floor((len(data) - 5) / (3 + math.ceil(num_links / 10)) / 15)
num_time_intervals = num_time_intervals if num_time_intervals % 2 == 0 else num_time_intervals - 1
vol_data_15 = [[]] * num_time_intervals
for index in tqdm(range(14, 14 + 15 * num_time_intervals ,15),desc='<1/2>'):
    vol_all = []
    for index_2 in range(math.ceil(num_links / 10)):
        vol_all = vol_all + [int(float(i)) for i in data[7 + (math.ceil(num_links / 10) + 3) * index + index_2].split()]
    vol_data_15[int((index - 14) / 15)] = vol_all
for index in range(num_time_intervals - 1,0,-1):
    vol_data_15[index] = [vol_data_15[index][i] - vol_data_15[index - 1][i] for i in range(0,num_links)]

vol_data = [[]] * int((num_time_intervals / 2))
for index in range(int((num_time_intervals / 2))):
    vol_data[index] = [(vol_data_15[index * 2][j] + vol_data_15[index * 2 + 1][j]) for j in range(len(vol_data_15[index * 2]))]

with open(pardir + r'/Preprocessed/vol_data.dat','wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(vol_data, filehandle)

#---------- get_VDF() -----------
vdf_param = pd.DataFrame(columns=['LinkID','t_0','Capacity'])
for index in tqdm(range(len(link_data)),desc='<2/2>'):
    vdf_param.loc[index] = pd.Series({'LinkID':int(index),
                                      't_0':int(link_data.loc[index]['Length']) / int(link_data.loc[index]['SpeedLimit']) / 5280 * 60,
                                      'Capacity':int(link_data.loc[index]['#Lanes']) * 2400})
print('Saving Files ...  ', end='',flush=True)
t_0_data = vdf_param['t_0'].tolist()
with open(pardir + r'/Preprocessed/t_0_data.dat','wb') as filehandle:
    pickle.dump(t_0_data, filehandle)

capacity_data = vdf_param['Capacity'].tolist()
with open(pardir + r'/Preprocessed/capacity_data.dat','wb') as filehandle:
    pickle.dump(capacity_data, filehandle)
print('Done')
print()
#----------------------------------- Data Organizing ------------------------------------
print('       Data Organizing       <step 5/5>          Processing... ')
#----- create_adjacent_list() ------
alpha = 0.15
beta = 4.0
adjListAll = [[] for _ in range(len(vol_data))]
for i in tqdm(range(len(vol_data)),desc='<1/8>'):
    adjList = [[] for _ in range(1 + len(nodes_mapping))]
    for j in range(len(link_data)):
        a_node = link_data.iloc[j]['FromID']
        b_node = link_data.iloc[j]['ToID']
        t_0 = vdf_param.iloc[j]['t_0']
        capacity = vdf_param.iloc[j]['Capacity']
        volume = vol_data[i][j]
        cost = t_0 * (1 + alpha * pow(volume * 2 / capacity, beta))
        adjList[a_node].append([b_node,cost])
    adjListAll[i] = adjList

with open(pardir+ r'/Preprocessed/adjListAll.dat', 'wb') as filehandle:
        # store the data as binary data stream
    pickle.dump(adjListAll, filehandle)

link_index_adjList_mapping = pd.DataFrame(columns=['Index', 'i', 'j', 'FromID', 'ToID'])

for i in tqdm(range(1, len(adjListAll[0])),desc='<2/8>'):
    for j in range(len(adjListAll[0][i])):
        index = link_data.index[(link_data['FromID'] == i) & (link_data['ToID'] == adjListAll[0][i][j][0])].values[0]
        link_index_adjList_mapping = link_index_adjList_mapping.append(
            {'Index': index, 'i': i, 'j': j, 'FromID': i, 'ToID': adjListAll[0][i][j][0]}, ignore_index=True)

link_index_adjList_mapping_i = link_index_adjList_mapping['i'].tolist()
with open(pardir + r'/Preprocessed/link_index_adjList_mapping_i.dat','wb') as filehandle:
    pickle.dump(link_index_adjList_mapping_i, filehandle)
link_index_adjList_mapping_j = link_index_adjList_mapping['j'].tolist()
with open(pardir + r'/Preprocessed/link_index_adjList_mapping_j.dat','wb') as filehandle:
    pickle.dump(link_index_adjList_mapping_j, filehandle)
link_index_adjList_mapping_fromid = link_index_adjList_mapping['FromID'].tolist()
with open(pardir + r'/Preprocessed/link_index_adjList_mapping_fromid.dat','wb') as filehandle:
    pickle.dump(link_index_adjList_mapping_fromid, filehandle)
link_index_adjList_mapping_toid = link_index_adjList_mapping['ToID'].tolist()
with open(pardir+ r'/Preprocessed/link_index_adjList_mapping_toid.dat','wb') as filehandle:
    pickle.dump(link_index_adjList_mapping_toid, filehandle)

#--- import_distance_matrix() ----
path = pardir + r'/Input/xy.dat'
with open(path, 'r') as file:
    data = file.readlines()
coordinate = [[int(data[index].split()[0]),
              float(data[index].split()[1]),
              float(data[index].split()[2])] for index in range(len(data))]
coordinate = pd.DataFrame(coordinate)
coordinate.columns = ['NodeID', 'X', 'Y']
#coordinate = coordinate.loc[coordinate['NodeID'] >= 500]
#coordinate = coordinate.reset_index(drop=True)

coordinate['TAZ_Centroid'] = 0
for i in tqdm(range(len(coordinate)),desc='<3/8>'):
    if len(nodes_mapping.loc[nodes_mapping['OriginalID'] == int(coordinate.iloc[i]['NodeID'])]['NewID']) == 0:
        coordinate.loc[i,'TAZ_Centroid'] = 1
        continue
    coordinate.loc[i,'NodeID'] = nodes_mapping.loc[nodes_mapping['OriginalID'] == int(coordinate.iloc[i]['NodeID'])]['NewID'].values[0]

coordinate = coordinate.loc[coordinate['TAZ_Centroid'] == 0]
coordinate = coordinate.drop(['TAZ_Centroid'], axis = 1)
coordinate = coordinate.reset_index(drop=True)

coordinate_1 = copy.deepcopy(coordinate)
coordinate_2 = copy.deepcopy(coordinate)
coordinate_1['key'] = 1
coordinate_2['key'] = 1
coordinate_combination = pd.merge(coordinate_1, coordinate_2, on='key').drop('key', axis=1)
coordinate_combination['X_diff'] = coordinate_combination['X_x'] - coordinate_combination['X_y']
coordinate_combination['X_diff'] = coordinate_combination['X_diff'] * coordinate_combination['X_diff']
coordinate_combination['Y_diff'] = coordinate_combination['Y_x'] - coordinate_combination['Y_y']
coordinate_combination['Y_diff'] = coordinate_combination['Y_diff'] * coordinate_combination['Y_diff']
coordinate_combination['Distance'] = coordinate_combination['X_diff'] + coordinate_combination['Y_diff']
coordinate_combination['Distance'] = pow(coordinate_combination['Distance'],0.5)
real_distance = [None] * len(coordinate_combination)
ratio = [None] * len(coordinate_combination)
for i in tqdm(range(len(link_data)),desc='<4/8>'):
    a_node = link_data.iloc[i]['FromID'] - 1
    b_node = link_data.iloc[i]['ToID'] - 1
    real_distance[a_node * len(coordinate) + b_node] = link_data.iloc[i]['Length']
    ratio[a_node * len(coordinate) + b_node] = coordinate_combination.iloc[a_node * len(coordinate) + b_node]['Distance'] / real_distance[a_node * len(coordinate) + b_node]
coordinate_combination['RealDistance'] = real_distance
coordinate_combination['Ratio'] = ratio

avg_ratio = sum(filter(None, ratio)) / len(link_data)

coordinate_combination['ConvertedDist'] = coordinate_combination['Distance'] / avg_ratio
coordinate_combination['Cost'] = coordinate_combination['ConvertedDist'] / 5280 / 45 * 60

rows, cols = (len(coordinate), len(coordinate))
dist_data = [[0] * cols] * rows

for i in tqdm(range(len(nodes_mapping)),desc='<5/8>'):
    dist_data[i] = coordinate_combination['Cost'][len(coordinate) * i:len(coordinate) * (i + 1)].tolist()

with open(pardir + r'/Preprocessed/dist_data.dat', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(dist_data, filehandle)

#--- import_link_data_index() ----

rows, cols = (len(nodes_mapping), len(nodes_mapping))
link_data_index = [[None] * cols for _ in range(rows)]
for i in tqdm(range(len(link_data)),desc='<6/8>'):
    link_data_index[link_data.iloc[i]['FromID'] - 1][link_data.iloc[i]['ToID'] - 1] = i

with open(pardir + r'/Preprocessed/link_data_index.dat', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(link_data_index, filehandle)

#--- original vehicle data -------

path = pardir + r'/Input/vehicle.dat'
with open(path, 'r') as file:
    data = file.readlines()

original_vehicle_data_row_1 = [data[index].split() for index in tqdm(range(2, len(data),2),desc='<7/8>')]
original_vehicle_data_row_1 = pd.DataFrame(original_vehicle_data_row_1)

original_vehicle_data_row_2 = [data[index + 1].split() for index in tqdm(range(2, len(data),2),desc='<8/8>')]
original_vehicle_data_row_2 = pd.DataFrame(original_vehicle_data_row_2)

original_vehicle_data = pd.concat([original_vehicle_data_row_1, original_vehicle_data_row_2], axis=1)

del original_vehicle_data_row_1, original_vehicle_data_row_2, data

print('Saving Files ...  ', end='',flush=True)

original_vehicle_data.columns = ['#','usec','dsec','Stime','vehcls','vehtype','ioc','#ONode','#IntDe','info','ribf','comp','izone','Evac',
                                 'InitPos','VoT','tFlag','pArrTime','TP','IniGas','DZone', 'waitTime']

original_vehicle_data.to_csv(pardir + r'/Preprocessed/original_vehicle_data.csv')

print('Done')
print()
