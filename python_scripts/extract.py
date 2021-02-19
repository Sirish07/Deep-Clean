'''
Extracting the csv_files in raw and pre-processed folders
'''

import os
import pickle

files = {}
sfile = {}
input_files = {}
output_files = {}

root = os.getcwd()
path = os.path.join(root,"scratch/Dataset/")

for r,d,filez in os.walk(path):
    if r.endswith('/EEG/raw/csv_format'):
        id = r.split('/')
        print(id)
        if id[-4] not in files:
            files[id[-4]] = []
        for f in sorted(filez):
            if '001.csv' in f and id[-4] not in sfile: 
                files[id[-4]].append(os.path.join(r,f))
                sfile[id[-4]] = f
    elif r.endswith('/EEG/preprocessed/csv_format'):
        id = r.split('/')
        print(id)
        if id[-4] not in files:
            files[id[-4]] = []
        for f in sorted(filez):
            if f.endswith(sfile[id[-4]]):
                files[id[-4]].append(os.path.join(r,f))

for k,v in files.items():
    if len(files[k])==2:
        input_files[k] = files[k][0]
        output_files[k] = files[k][1]

for k,v in input_files.items():
    x = input_files[k]
    y = output_files[k]
    x = x.rsplit('/')
    y = y.rsplit('/')
    print(k,x[-1],y[-1])

with open('finish.p', 'wb') as fp:
    pickle.dump(finished, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('Train_data.p','wb') as f:
    pickle.dump(input_files,f,protocol=pickle.HIGHEST_PROTOCOL)
    
with open('Train_truth.p', 'wb') as fp:
    pickle.dump(output_files, fp, protocol=pickle.HIGHEST_PROTOCOL)


'''
Extracting the EOG and EEG components
'''

import os

root = os.getcwd()
inputp = os.path.join(root,"projects/def-kjerbi/sirish01/TrainData/")
ifiles = sorted(os.listdir(inputp))

subject_names = []
infiles = []

for x in ifiles:
    infiles.append(os.path.join(inputp,x))
    subject_names.append(x[:-10])

def Diff(li1, li2): 
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

EOG = [7,13,16,20,24,124,125,126,127]
channels = np.arange(128)
EEG = Diff(channels,EOG)

save1 = os.path.join(root,"projects/def-kjerbi/sirish01/TrainData/EOGData/")
save2 = os.path.join(root,"projects/def-kjerbi/sirish01/TrainData/EEGData/")

if not os.path.isdir(save1):
    os.makedirs(save1)
if not os.path.isdir(save2):
    os.makedirs(save2)

for i in range(len(infiles)):
    data = np.load(infiles[i])
    EOG_data = data[:,:,EOG]
    EEG_data = data[:,:,EEG]
    savefile1 = save1+str(subject_names[i])+'_EOG.npy'
    savefile2 = save2+str(subject_names[i])+'_EEG.npy'
    print(EOG_data.shape,EEG_data.shape)
    np.save(savefile1,EOG_data)
    np.save(savefile2,EEG_data) 