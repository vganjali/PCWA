import numpy as np
import pandas as pd
import pcwa as pcwa
import matplotlib.pyplot as plt
# read the raw mass scpectroscopy data and truth values
# df_raw = pd.read_csv('n100sig66_dataset_1_25/Dataset_7/RawSpectra/noisy7.txt',sep=' ')
# df_true = pd.read_csv('n100sig66_dataset_1_25/Dataset_7/truePeaks/truth7.txt',sep=' ')
df_raw = pd.read_csv('n100sig66_dataset_1_25/Dataset_14/RawSpectra/noisy22.txt',sep=' ')
df_true = pd.read_csv('n100sig66_dataset_1_25/Dataset_14/truePeaks/truth22.txt',sep=' ')

# create pcwa_analyzer object and set the desired parameters
pcwa_analyzer = pcwa.PCWA()
pcwa_analyzer.trace = df_raw['Intensity']
pcwa_analyzer.dt = 1
pcwa_analyzer.scales = [10,100,100]
pcwa_analyzer.wavelet = ['ricker']
pcwa_analyzer.keep_cwt = False
pcwa_analyzer.w, pcwa_analyzer.h = 0.2, 1
pcwa_analyzer.show_wavelets = False
pcwa_analyzer.usescratchfile = False

# detect events (peaks)
events = pcwa_analyzer.detect_events(threshold=200)
print(len(events))
# fine tune the location of detected peaks
loc = [int(e-events['scale'][n]+np.argmax(df_raw['Intensity'][int(e-events['scale'][n]):int(e+events['scale'][n])])) for n,e in enumerate(events['time'])]
print(len(loc))

# plot the raw data, true peaks and detected peaks
fig, ax = plt.subplots(3,1,figsize=(16,4),dpi=96,sharex=True,gridspec_kw={'height_ratios': [12,1,1]})
plt.subplots_adjust(hspace=0,wspace=0)
l0, = ax[1].plot(df_true['Mass'],df_true['Particles']*0, '|',markersize=10,color='gray',label='Truth')
ax[0].plot(df_raw['Mass'],df_raw['Intensity'],color='blue')
l1, = ax[2].plot(df_raw['Mass'].iloc[loc],[0]*len(loc),'|',markersize=10,color='red',label='PCWA')
# l2, = ax[0].plot(df_raw['Mass'].iloc[loc],df_raw['Intensity'].iloc[loc],'ro',markerfacecolor='none',markersize=10)
ax[1].set_yticks([])
ax[1].set_ylim(0,0)
ax[2].set_yticks([])
ax[2].set_ylim(0,0)
ax[0].set_ylabel('Intensity')
ax[-1].set_xlabel('m/z')
plt.legend(handles=[l0,l1], bbox_to_anchor=(1.0, 4), loc='upper left')
plt.show()
# print(events)
# calculate and print TPR and FDR values
true_peaks = np.sort(df_true['Mass'].to_numpy())
detected_peaks = np.sort(df_raw['Mass'].iloc[loc].to_numpy())
tpr, fdr = pcwa.tprfdr(true_peaks, detected_peaks, e=0.01, MS=True)
print(f"TPR={tpr:.3f}, FDR={fdr:.3f}")