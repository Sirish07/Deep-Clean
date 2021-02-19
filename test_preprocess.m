file = '/home/sirish001/Documents/Acads/Dataset/A00051826/EEG/raw/raw_format/A00051826001.raw';

data = pop_readegi(file);
%%
display(data)

load_ch = load('chan111.mat');
channels = load_ch.chan111;
eog_channels = sort([1 32 8 14 17 21 25 125 126 127 128]);
data.data(end+1,:) = 0;
data.nbchan = data.nbchan + 1;
data = pop_chanedit(data,  ...
    'load',{ ['GSN-HydroCel-129.sfp'], 'filetype', 'sfp'});

clear chan111
s = size(data.data);
assert(data.nbchan == s(1)); 

clear s;

%% Preprocess data
% filtering on the whole dataset
rawdata = data.data;
fprintf("Raw data range is\n");
min(min(rawdata)), max(max(rawdata))

filtered_data = perform_filter(data,'US');
fprintf("Filtered data range is \n");
min(min(filtered_data.data)), max(max(filtered_data.data))

% seperate EEG channels from EOG ones
unique_chans = setdiff(channels, eog_channels);
[~, EEG] = evalc('pop_select( filtered_data , ''channel'', channels)');
[~, EOG] = evalc('pop_select( filtered_data , ''channel'', eog_channels)');
[~, EEG_unique] = evalc('pop_select( filtered_data , ''channel'', unique_chans)');

% Detect artifact channels
rejected_chans = reject_channels(EEG);

% Remove effect of EOG
EEG_regressed = EOG_regression(EEG_unique, EOG);
fprintf("EEG_regressed data range is \n");
min(min(EEG_regressed.data)), max(max(EEG_regressed.data))
%%
% % PCA, MARA
[EEG_cleared, noise] = perform_pca(EEG_regressed);
% fprintf("Performing MARA\n");
% size(EEG_regressed.data)
% 
% [EEG_cleared] = processMARA([EEG_regressed],EEG_regressed,0,[1 1 1 1 1]);
fprintf("PCA/MARA cleaned data range is \n")
min(min(EEG_cleared.data)), max(max(EEG_cleared.data))
% Replace common channels of eeg and eog by zeros and put together with
% unique eeg channels
eeg_cleaned = zeros(size(filtered_data.data));
eeg_cleaned(unique_chans,:) = EEG_cleared.data;
EEG_cleaned = EEG;
EEG_cleaned.data = eeg_cleaned(channels,:);

% interpolate zero and artifact channels:
display('Interpolating...');
if ~isempty(rejected_chans)
    [~, interpolated] = evalc('eeg_interp(EEG_cleaned ,rejected_chans ,''spherical'')');
end
interpolated.auto_badchans = rejected_chans;
% detrending
res = bsxfun(@minus, interpolated.data, mean(interpolated.data,2));
result = interpolated;
result.data = res;

preprocessed_data = res;
fprintf("Interpolated data range is \n")
min(min(res)), max(max(res))