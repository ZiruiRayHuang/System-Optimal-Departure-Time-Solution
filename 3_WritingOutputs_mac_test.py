import pickle
import pandas as pd
import os
import sys
from tqdm import tqdm

#------------------------------------ DEFINE FUNCTIONS -----------------------------------

def new_id_to_original_id(new_id, mapping):
    #if new_id >= 5359:
    #    return (new_id + 298)
    #elif new_id >= 5345:
    #    return (new_id + 297)
    #elif new_id >= 4989:
    #    return (new_id + 296)
    #elif new_id >= 507:
    #    return (new_id + 295)
    #else:
    #    return (new_id + 294)
    mapping = [[item[0] - item[1],item[1]] for item in mapping]
    for i in range(len(mapping)):
        if new_id <= mapping[i][0]:
            return new_id + mapping[i][1]


def new_path_list_to_original_string(path_list, mapping):
    path_list = [new_id_to_original_id(path_list[i], mapping) for i in range(len(path_list))]
    line = "{:>7s}" * len(path_list) + "\n"
    line = line.format(*[str(path_list[i]) for i in range(len(path_list))])
    return line

#------------------------------------ WRITING OUTPUTS  -----------------------------------
pardir = os.path.abspath(os.path.join(os.path.realpath(sys.executable),os.path.pardir,os.path.pardir))
if os.path.exists(pardir + r'/Output/'):
    pass
else:
    os.mkdir(pardir + r'/Output/')

print()
print()
print()
print('****************** PART III: WRITING OUTPUTS *****************')
print()

path = pardir + r'/Input/config.dat'
with open(path, 'r') as file:
    data = file.readlines()
start_time = int(data[0].rstrip().split()[0])
end_time = int(data[0].rstrip().split()[1])
peak = [[] for _ in range(len(data) - 3)]
for i in range(len(data) - 3):
    peak[i] = [int(data[1 + i].rstrip().split()[0]), int(data[1 + i].rstrip().split()[1])]
participate_rate = float(data[-2].rstrip()) / 100
max_permissive_hour = int(data[-1].rstrip())


with open(pardir + r'/Output/path_data.dat', 'rb') as filehandle:
    path_data = pickle.load(filehandle)

with open(pardir + r'/Output/shift_data.dat', 'rb') as filehandle:
    shift_data = pickle.load(filehandle)

with open(pardir + r'/Preprocessed/vehicle_data', 'rb') as filehandle:
    vehicle_data = pickle.load(filehandle)

path = pardir + r'/Input/path.dat'
with open(path, 'r') as file:
    original_path_data = file.readlines()

path = pardir + r'/Input/vehicle.dat'
with open(path, 'r') as file:
    data = file.readlines()

vehicle_first_row = data[0]

original_vehicle_data = pd.read_csv(pardir + r'/Preprocessed/original_vehicle_data.csv')

with open(pardir + r'/Preprocessed/mapping.dat', 'rb') as filehandle:
    mapping = pickle.load(filehandle)

Stime_shift = [(lambda i: shift_data[i][0][0] * 30 if shift_data[i] != [] else 0)(i) for i in tqdm(range(len(vehicle_data)),desc='<1/4>')]
vehicle_data['Stime_shift'] = Stime_shift
vehicle_data['Stime'] = vehicle_data['Stime'] + vehicle_data['Stime_shift']
path_data = [(lambda i: new_path_list_to_original_string(path_data[i], mapping) if path_data[i] != [] else original_path_data[i])(i) for i in tqdm(range(len(path_data)),desc='<2/4>')]
vehicle_data['path'] = path_data


original_vehicle_data['path'] = original_path_data

original_vehicle_data.loc[(original_vehicle_data['Stime'] >= start_time) & (original_vehicle_data['Stime'] < end_time), 'path'] = path_data
original_vehicle_data.loc[(original_vehicle_data['Stime'] >= start_time) & (original_vehicle_data['Stime'] < end_time), 'Stime'] = vehicle_data['Stime']

original_vehicle_data['#ONode'] = [len(item.split()) for item in original_vehicle_data['path'].tolist()]
original_vehicle_data['OriginalVehID'] = range(1,len(original_vehicle_data) + 1)

original_vehicle_data = original_vehicle_data.sort_values(by=['Stime','#'])
original_vehicle_data['#'] = range(1,len(original_vehicle_data) + 1)
original_vehicle_data = original_vehicle_data.reset_index(drop=True)

new_vehicle_data_VehID = original_vehicle_data['#'].tolist()
new_vehicle_data_Usec = original_vehicle_data['usec'].tolist()
new_vehicle_data_Dsec = original_vehicle_data['dsec'].tolist()
new_vehicle_data_Stime = original_vehicle_data['Stime'].tolist()
new_vehicle_data_Vehcls = original_vehicle_data['vehcls'].tolist()
new_vehicle_data_Vehtype = original_vehicle_data['vehtype'].tolist()
new_vehicle_data_loc = original_vehicle_data['ioc'].tolist()
new_vehicle_data_ONode = original_vehicle_data['#ONode'].tolist()
new_vehicle_data_IntDe = original_vehicle_data['#IntDe'].tolist()
new_vehicle_data_Info = original_vehicle_data['info'].tolist()
new_vehicle_data_Ribf = original_vehicle_data['ribf'].tolist()
new_vehicle_data_Comp = original_vehicle_data['comp'].tolist()
new_vehicle_data_Izone = original_vehicle_data['izone'].tolist()
new_vehicle_data_Evac = original_vehicle_data['Evac'].tolist()
new_vehicle_data_InitPos = original_vehicle_data['InitPos'].tolist()
new_vehicle_data_VoT = original_vehicle_data['VoT'].tolist()
new_vehicle_data_tFlag = original_vehicle_data['tFlag'].tolist()
new_vehicle_data_pArrTime = original_vehicle_data['pArrTime'].tolist()
new_vehicle_data_TripPurpose = original_vehicle_data['TP'].tolist()
new_vehicle_data_InitialGas = original_vehicle_data['IniGas'].tolist()
new_vehicle_data_DZone = original_vehicle_data['DZone'].tolist()
new_vehicle_data_waitTime = original_vehicle_data['waitTime'].tolist()

new_vehicle_data_row_1 = [('{:>9s}{:>7s}{:>7s}{:>8s}{:>6s}{:>6s}{:>6s}{:>6s}{:>6s}{:>6s}{:>8s}{:>8s}{:>5s}{:>5s}{:>12s}{:>8s}{:>5s}{:>7s}{:>5s}{:>5s}\n' \
                            .format(str(int(new_vehicle_data_VehID[i])),
                                    str(int(new_vehicle_data_Usec[i])),
                                    str(int(new_vehicle_data_Dsec[i])),
                                    "%.2f" % new_vehicle_data_Stime[i],
                                    str(int(new_vehicle_data_Vehcls[i])),
                                    str(int(new_vehicle_data_Vehtype[i])),
                                    str(int(new_vehicle_data_loc[i])),
                                    str(int(new_vehicle_data_ONode[i])),
                                    str(int(new_vehicle_data_IntDe[i])),
                                    str(int(new_vehicle_data_Info[i])),
                                    "%.4f" % new_vehicle_data_Ribf[i],
                                    "%.4f" % new_vehicle_data_Comp[i],
                                    str(int(new_vehicle_data_Izone[i])),
                                    str(int(new_vehicle_data_Evac[i])),
                                    "%.8f" % new_vehicle_data_InitPos[i],
                                    "%.2f" % new_vehicle_data_VoT[i],
                                    str(int(new_vehicle_data_tFlag[i])),
                                    "%.1f" % new_vehicle_data_pArrTime[i],
                                    str(int(new_vehicle_data_TripPurpose[i])),
                                    "%.1f" % new_vehicle_data_InitialGas[i])) for i in tqdm(range(len(original_vehicle_data)),desc='<3/4>')]

new_vehicle_data_row_2 = [('{:>12s}{:>7s}\n'.format(str(int(new_vehicle_data_DZone[i])),"%.2f" % new_vehicle_data_waitTime[i])) for i in tqdm(range(len(original_vehicle_data)),desc='<4/4>')]

original_vehicle_data['Row1'] = new_vehicle_data_row_1
original_vehicle_data['Row2'] = new_vehicle_data_row_2

new_vehicle_data_row = [None]*(len(new_vehicle_data_row_1) + len(new_vehicle_data_row_2))
new_vehicle_data_row[::2] = new_vehicle_data_row_1
new_vehicle_data_row[1::2] = new_vehicle_data_row_2

new_data = [[]] * (2 + 2 * len(original_vehicle_data))
new_data[0] = vehicle_first_row
new_data[1] = '        #   usec   dsec   stime vehcls vehtype ioc #ONode #IntDe info ribf    comp   izone Evac InitPos    VoT  tFlag pArrTime TP IniGas\n'
new_data[2:len(new_data)] = new_vehicle_data_row

print()

print('Saving Files ...  ', end='',flush=True)

with open(pardir + r'/Output/output_vehicle.dat', 'w') as file:
    file.writelines(new_data)

with open(pardir + r'/Output/output_path.dat', 'w') as file:
    file.writelines(original_vehicle_data['path'].tolist())

os.remove(pardir + r'/Output/path_data.dat')
os.remove(pardir + r'/Output/shift_data.dat')

print('Done')
print()

print("* ADJUSTMENTS FINISHED. PLEASE CHECK RESULTS UNDER 'OUTPUT' FOLDER *")
