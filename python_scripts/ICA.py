import os
import mne
import numpy as np
from mne.preprocessing import ICA

root = os.getcwd()
inputpath = os.path.join(root,"projects/def-kjerbi/sirish01/TrainData/")
save_loc = os.path.join(root,"scratch/models/unknown/tmp/ICA/")
if not os.path.isdir(save_loc):
	os.makedirs(save_loc)

infiles = os.listdir(inputpath) 
infiles = sorted(infiles)
subject_names = []
for x in infiles:
	subject_names.append(x[:-10])
inpaths = []

def apply_ICA(data,id):
	sfreq = 100
	times = np.arange(0,data.shape[1])
	ch_types = []
	ch_names = []
	for i in range(data.shape[0]):
		ch_types.append('eeg')
		ch_names.append('Factors')
	info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
	raw = mne.io.RawArray(data, info)
	ica = ICA(n_components = 30,max_pca_components = 30,method = 'fastica')
	ica.fit(raw)
	source_signals = ica.get_sources(raw)
	source_data = source_signals._data
	save_file = save_loc + str(id) + "_ICA.npy"
	np.save(save_file,source_data)

for x in infiles:
	inpaths.append(os.path.join(inputpath,x))

for i in range(len(inpaths)):
	t = np.load(inpaths[i])
	data = np.zeros((t.shape[2],t.shape[0]*t.shape[1]))
	for j in range(t.shape[0]):
		for k in range(50):
			data[:,j*40+k] = t[j,k,:]
	apply_ICA(data,subject_names[i])

