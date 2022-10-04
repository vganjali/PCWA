import sys
sys.path.append('../PCWA')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.pcwa import pcwa

# read the raw mass scpectroscopy data and truth values
df_raw = pd.read_csv('tests/n100sig66_dataset_1_25/Dataset_14/RawSpectra/noisy22.txt',sep=' ')
df_true = pd.read_csv('tests/n100sig66_dataset_1_25/Dataset_14/truePeaks/truth22.txt',sep=' ')

# create pcwa_analyzer object and set the desired parameters
pcwa_analyzer = pcwa.PCWA()
pcwa_analyzer.trace = df_raw['Intensity']
pcwa_analyzer.dt = 1
pcwa_analyzer.scales_range = [10,1000,500]
# pcwa_analyzer.events_scales_range = [10,100]
pcwa_analyzer.wavelet = ['ricker']
pcwa_analyzer.keep_cwt = True
pcwa_analyzer.w, pcwa_analyzer.h = 1, 1
pcwa_analyzer.show_wavelets = False
pcwa_analyzer.use_scratchfile = False

# detect events (peaks)
events = pcwa_analyzer.detect_events(threshold=200)

# fine tune the location of detected peaks
loc = events['loc']
# loc = [int(e-events['scale'][n]+np.argmax(df_raw['Intensity'][int(e-events['scale'][n]):int(e+events['scale'][n])])) for n,e in enumerate(events['loc'])]

true_peaks = np.sort(df_true['Mass'].to_numpy())
detected_peaks = np.sort(df_raw['Mass'].iloc[loc].to_numpy())
tpr, fdr = pcwa.tprfdr(true_peaks, detected_peaks, e=0.01, MS=True)
print(f"TPR={tpr:.3f}, FDR={fdr:.3f}")

fig, ax = plt.subplots(3,1,figsize=(16,4),dpi=96,sharex=True,gridspec_kw={'height_ratios': [12,1,1]})
l0, = ax[1].plot(df_true['Mass'],df_true['Particles']*0, '|',markersize=10,color='gray',label='Truth')
ax[0].plot(df_raw['Mass'],df_raw['Intensity'],color='blue',lw=0.5)
# ax[0].scatter(df_raw['Mass'].iloc[loc],df_raw['Intensity'].iloc[loc],color='red',marker='o',s=10,facecolors='none',zorder=10)
ax[0].scatter(df_raw['Mass'].iloc[loc],events['coeff'],color='red',marker='o',s=10,facecolors='none',zorder=10)
l1, = ax[2].plot(df_raw['Mass'].iloc[loc],[0]*len(loc),'|',markersize=10,color='red',label='PCWA')
ax[1].set_yticks([])
ax[1].set_ylim(-1e-12,1e-12)
ax[2].set_yticks([])
ax[2].set_ylim(-1e-12,1e-12)
ax[0].set_ylabel('Intensity')
ax[-1].set_xlabel('m/z')
ax[0].legend(handles=[l0,l1], loc='upper right')
plt.tight_layout()
plt.subplots_adjust(hspace=0,wspace=0)
plt.show()
if pcwa_analyzer.keep_cwt:
    fig, ax = plt.subplots(1,1,figsize=(16,4))
    _ax = plt.twinx(ax)
    _ax.set_zorder(-1)
    _ax.imshow(pcwa_analyzer.cwt['ricker'].T,extent=[0,pcwa_analyzer.cwt['ricker'].shape[0],\
        pcwa_analyzer.scales_range[0],pcwa_analyzer.scales_range[1]],aspect='auto',origin='lower',cmap='hot')
    _ax.set_yticks([])
    ax.scatter(events['loc'],events['scale'],color='white',marker='o',facecolors='none')
    ax.set_facecolor('none')
    ax.set_yscale('log')
    ax.set_ylim(pcwa_analyzer.scales_range[0],pcwa_analyzer.scales_range[1])
    plt.tight_layout()
    plt.show()