import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import pickle
import mne
import datetime
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pdb

root = os.getcwd()
saveo = os.path.join(root,"projects/def-kjerbi/sirish01/TrainTruth/")

outputpath = os.path.join(root,"Train_truth.p")
inputpath = os.path.join(root,"Train_data.p")
complete = 'finish.p'

def resample(data):
	sfreq = 500 #for our case
	times = np.arange(0,data.shape[1])
	ch_types = []
	ch_names = []
	for i in range(data.shape[0]):
		ch_types.append('eeg')
		ch_names.append('Data')
	info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
	raw = mne.io.RawArray(data, info)
	raw_downsampled = raw.copy().resample(sfreq=100) #required
	downsampled_data = raw_downsampled._data
	newdata = pd.DataFrame(downsampled_data)
	return newdata

def preprocess(id,f,input = False):
	chunksize = 1000000
	df_list = []
	for df_chunk in tqdm(pd.read_csv(f,chunksize=chunksize, header = None)):
		df_list.append(df_chunk) 
	t = pd.concat(df_list)
	del df_list
	del df_chunk
	y = resample(t)
	window = 50
	shift = 40 #overlap = 20%
	target_data = []
	for i in range(len(y)):
		t = []
		idx = 0
		while True:
			end = idx*shift+window-1
			if end>=len(y.loc[i,:]):
				break
			l = y.loc[i,idx*shift:end]
			t.append(l)
			idx += 1
		target_data.append(t)
	p = len(target_data)
	q = len(target_data[0])
	r = window
	print(p,q,r)
	x = np.zeros((p,q,r)).astype('float')
	for i in range(p):
		x[i,:,:] = np.array(target_data[i])
	result = np.zeros((q,r,p)).astype('float')
	for i in range(q):
		for j in range(r):
			result[i,j,:] = x[:,i,j]
	if input==True:
		savefilepath = savei + str(id) + "_input.npy"
	else:
		savefilepath = saveo + str(id) + "_target.npy"
	print(id,result.shape,x.min(),x.max())
	np.save(savefilepath,result)
	
with open(complete,'rb') as f:
	finish = pickle.load(f)

with open(outputpath,'rb') as f:
	output = pickle.load(f)


