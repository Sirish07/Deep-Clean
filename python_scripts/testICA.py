import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from autoreject import get_rejection_threshold 

def Diff(li1, li2): 
	return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
'''
Things to do are:
1. Extracting the events out of the data successfully
2. Observe why the ranges in the raw data are different from .csv file
3. extract epochs and events using the .raw file.
4, Calculate the global reject threshold
5. apply ICA algorithm
6. identify eog_epochs
'''
eog_channels = [0, 31, 7, 13, 16, 20, 24, 124, 125, 126, 127]
channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 128]
unique_channels = Diff(channels, eog_channels)

file = 'A00051826001.raw'
A = mne.io.read_raw_egi(file, eog = eog_channels, preload = True)
A.filter(1., 40., n_jobs=-1, fir_design='firwin')
events = mne.find_events(A)
picks = mne.pick_types(A.info, meg=False, eeg=True, eog=True,
                           stim=False)
eegdata,_ = A[channels]
eogdata,_ = A[eog_channels]
eeg_unique,_ = A[unique_channels]


'''
Artifact rejection using regression
'''
tmin, tmax = -0.2, 0.5
event_id = {'Open': 1, 'Close' : 2}
epochs = mne.Epochs(A, events, event_id, tmin, tmax, 
                    picks=picks, baseline=(None, None), preload=True,
                    reject=None, verbose=True, detrend=0)
_, betas = mne.preprocessing.regress_artifact(epochs.copy().subtract_evoked())

# A, _ = mne.preprocessing.regress_artifact(A, betas=betas)
epochs_clean, _ = mne.preprocessing.regress_artifact(epochs, betas=betas)


reject = get_rejection_threshold(epochs_clean)



n_components = 25
method = 'fastica' 
decim = 1
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state)
ica.fit(A, decim=decim)

eog_average = create_eog_epochs(A, reject=reject).average()
eog_epochs = create_eog_epochs(A) 
eog_inds, scores = ica.find_bads_eog(A)


ica.exclude.extend(eog_inds)
