import os
import numpy as np
import torch


fpath = '/home/sirish01/projects/def-kjerbi/sirish01/models/best/30_epochs/10-factors'
ipath = '/home/sirish01/projects/def-kjerbi/sirish01/Data/Processed/EOGData'
save_loc = '/home/sirish01/projects/def-kjerbi/sirish01/models/best/30_epochs/10-factors/EOG_Corr/")

if not os.path.isdir(save_loc):
	os.makedirs(save_loc)

fsub = sorted(os.listdir(fpath))
isub = sorted(os.listdir(ipath))

s1 = []
for x in isub:
	s1.append(x[:-8])

s2 = []
for x in fsub:
	if x.endswith('.pth')
	s2.append(x[:-16])

print(s1==s2)


fdata = []
idata = []
for x in fsub:
	if x.endswith('.pth'):
		fdata.append(os.path.join(fpath,x))

for x in isub:
	idata.append(os.path.join(ipath,x))

def correlation(x1,x2):
	return np.corrcoef(x1.flatten(), x2.flatten())[0,1]

for i in range(len(idata)):
	time_data = torch.load(fdata[i],map_location = torch.device('cpu'))
	f = time_data['time_series']['factors']
	factors = np.zeros((f.shape[2],f.shape[0]*f.shape[1])).astype('float')
	for j in range(f.shape[0]):
	    for k in range(50):
	        factors[:,j*40+k] = f[j,k,:]
	EOG = np.load(idata[i])
	EOG_data = np.zeros((EOG.shape[2],EOG.shape[0]*EOG.shape[1])).astype('float')
	for j in range(EOG.shape[0]):
	    for k in range(50):
	    	EOG_data[:,j*40+k] = EOG[j,k,:]
	Corrmatrix = np.zeros((factors.shape[0],EOG_data.shape[0]))
	for j in range(Corrmatrix.shape[0]):
		for k in range(Corrmatrix.shape[1]):
			Corrmatrix[j, k] = correlation(factors[j,:],EOG_data[k,:])
	savefile = save_loc + str(s1[i]) + '_f_cor.npy'
	print(factors.shape,EOG_data.shape)
	np.save(savefile,Corrmatrix)
	

