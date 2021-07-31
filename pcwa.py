import numpy as np
import h5py
from scipy.special import erf
from scipy.special import erf
from scipy.signal import find_peaks, convolve
from math import floor, ceil
import time
import matplotlib.pyplot as plt
import multiprocessing as mp


#================================= Wavelet functions =================================#
def ricker(scale=10, N=1, window=1, dt=1):
    resolution = scale/dt
#     print(resolution)
    length = int((10*window)*resolution)
    a = resolution/1.25187536
    t = np.arange(length)
    s = 2/(np.sqrt(3*a)*np.pi**1/4)*(1-(t-length/2)**2/a**2)\
        *np.exp(-(t-length/2)**2/(2*a**2))
    s_square_norm = np.trapz(s**2, dx=1)
    s -= np.mean(s)
#     return s*(s_square_norm**2)
    return s/np.sqrt(s_square_norm)

def morlet(scale=10, N=6, window=1, is_complex=False, dt=1):
    resolution = scale/dt
    length = int(2*(N+4)*window*resolution)
    t = np.arange(length)
    sigma = length/(10*window)
    s_exp = np.exp(-(t-length/2)**2/(2*sigma**2))
    if (is_complex):
        s_sin = np.exp(1j*(2*np.pi/resolution*(t-length/2)-np.pi*(0.75-N%2)))
    else:
        s_sin = np.sin((2*np.pi/resolution*(t-length/2)-np.pi*(0.5-N%2)))
    s = s_exp*s_sin
    s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    return s/np.sqrt(s_square_norm)

def skew_normal(x, mu, sigma, alpha=0):
    # mean = mu - sigma*alpha/np.sqrt(1+alpha**2)*np.sqrt(2/np.pi)
    delta = alpha/(np.sqrt(1+alpha**2))
    mu_z = np.sqrt(2/np.pi)*delta
    sigma_z = np.sqrt(1-mu_z**2)
    gamma_1 = (4-np.pi)/2*(delta*np.sqrt(2/np.pi))**3/((1-2*delta**2/np.pi)**(3/2))
    if alpha == 0:
        m_0 = 0
    else:
        m_0 = mu_z - gamma_1*sigma_z/2 - np.sign(alpha)/2*np.exp(-2*np.pi/np.abs(alpha))
    mode = mu + sigma*m_0
    xi = mu - sigma*m_0
    phi = 1/np.sqrt(2*np.pi)*np.exp(-((x-xi)**2)/(2*sigma**2))
    _PHI = 1/2*(1+erf(alpha*(x-xi)/sigma/np.sqrt(2)))
    return 2/sigma*phi*_PHI

def msg(scale=10, N=6, pattern='6', window=1, mplx_ratio=1, weight=1, mod=0.5, shift=1, skewness=1, is_complex=False, dt=1, mf=False):
    N = int(N)
    if (type(N) != list):
        N = [N]
    if ((type(mplx_ratio) == float) or (type(mplx_ratio) == int)):
        mplx_ratio = [mplx_ratio]*len(N)
    elif (len(mplx_ratio) != len(N)):
        print('multiplex ratio should have same length as N')
        return
    if ((type(weight) == float) or (type(weight) == int)):
        weight = [weight]*max(N)
    elif (len(weight) != N):
        print('weight ratio should have a length equal to N')
        return
    N_max = max(N)
#     mod *= (4.29193/2.35482)
    skewness *= 1.2*mod
    resolution = scale/dt
    sigma = resolution/4.29193*mod*N_max/8
#     amp = np.array([3,4,3,2,4,3,3,3])
#     amp = amp[::-1]/np.sum(amp)*N_max
    # sigma = resolution/4*mod
    length = int(N_max*resolution + shift*resolution + (1+4*skewness)*5*N_max*window*sigma)
    t = np.arange(length)
    if is_complex:
        s = np.zeros((length,),dtype=np.complex)
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*weight[m]*skew_normal(t,length/2+((n-1)/2-m+0.125)*(N_max/n)*resolution,sigma, alpha=0)
                s += 1j*mplx_ratio[i]*skew_normal(t,length/2+((n-1)/2-m-0.125)*(N_max/n)*resolution,sigma, alpha=0)
        s -= np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1j*np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= 1j*np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
    else:
        s = np.zeros((length,))
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*weight[m]*skew_normal(t,length/2-((n-1)/2-m)*(N_max/n)*resolution,sigma, alpha=0)
    if not mf:
        s -= np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
    return s

def msg_encoded(scale=10, pattern='1201', window=1, mod=0.5, shift=1, skewness=1, is_complex=False, dt=1, mf=False):
    if (type(pattern) != list):
        pattern = [pattern]
    N = [0]*len(pattern)
    for i,n in enumerate(pattern):
        pattern[i] = [float.fromhex('0x'+p) for p in pattern[i]]
        N[i] = len(pattern[i])
    N_max = max(N)
    skewness *= 1.2*mod
    resolution = scale/dt
    sigma = resolution/4.29193*mod*N_max/8
    length = int(N_max*resolution + shift*resolution + (1+4*skewness)*5*N_max*window*sigma)
    t = np.arange(length)
    if is_complex:
        s = np.zeros((length,),dtype=np.complex)
        for i,n in enumerate(N):
            for m in range(n):
                s += float.fromhex('0x'+n[m])*skew_normal(t,length/2+((n-1)/2-m+0.125)*(N_max/n)*resolution,sigma, alpha=0)
                s += 1j*skew_normal(t,length/2+((n-1)/2-m-0.125)*(N_max/n)*resolution,sigma, alpha=0)
        s -= 1/2*skew_normal(t, length/2+(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1j/2*skew_normal(t, length/2+(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1/2*skew_normal(t, length/2-(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= 1j/2*skew_normal(t, length/2-(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
    else:
        s = np.zeros((length,))
        for i,n in enumerate(N):
            for m in range(n):
                s += pattern[i][m]*skew_normal(t,length/2-((n-1)/2-m)*(N_max/n)*resolution,sigma, alpha=0)
    if not mf:
        s -= np.sum(pattern)/2*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(pattern)/2*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
    return s

#============================== CWT and Event Detection ==============================#
d_type = np.dtype([('time', 'f8'), ('scale', 'f8'), ('coeff', 'f8'), ('N', 'f8')])
def cwt(data, scales, wavelets, use_scratch=True, show_wavelets=False):
    wvlts = {wavelet:{} for wavelet in wavelets}
    if use_scratch:
        try:
            _cwt = h5py.File("cwt.scratch", "w")
        except:
            _cwt = h5py.File("cwt.scratch", "r")
            _cwt.close()
            _cwt = h5py.File("cwt.scratch", "w")
    else:
        _cwt = {wavelet:[] for wavelet in wavelets}
    if show_wavelets:
        plt.figure()
    for wavelet in wavelets:
        if wavelet == 'ricker': 
            N = 1 
            wvlts[wavelet] = [ricker(s) for s in scales]
        elif wavelet[:4] == 'msg-':
            N = int(wavelet[4:]) 
            wvlts[wavelet] = {'N':N, 'w':[msg(s, N=N, mod=0.8, shift=1, skewness=0.5) for s in scales]} 
        elif wavelet[:5] == 'msge-':
            N = len(wavelet[5:]) 
            wvlts[wavelet] = {'N':N, 'w':[msg_encoded(s, pattern=wavelet[5:], mod=1.5, shift=-2.9, skewness=0.04) for s in scales]}
        elif wavelet[:7] == 'morlet-':
            N = int(wavelet[7:]) 
            wvlts[wavelet] = {'N':N, 'w':[morlet(s, N=N, is_complex=False) for s in scales]}
        elif wavelet[:8] == 'cmorlet-':
            N = int(wavelet[8:])
    #         print(N)
            wvlts[wavelet] = {'N':N, 'w':[morlet(s, N=N, is_complex=True) for s in scales]}
        if show_wavelets:
            plt.plot(wvlts[wavelet]['w'][0],label=wavelet)
#         print(len(wvlts[wavelet]['w'][0]))
        if use_scratch:
            _cwt.create_dataset(wavelet, (len(data),len(wvlts[wavelet]['w'])), chunks=True, dtype='float', compression="gzip")
            if np.iscomplexobj(wvlts[wavelet]['w'][0]):
                for n, w in enumerate(wvlts[wavelet]['w']):
                    _l = floor(min(len(data),len(w))/2)
                    _r = min(len(data),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = np.abs(convolve(data, w, mode='valid')) 
        #             _cwt[:,n] = (0.5*np.correlate(data, w, mode='same')) 
        #             _cwt[:,n] += np.abs(_cwt[:,n])      
            else:
                for n, w in enumerate(wvlts[wavelet]['w']):
                    _l = floor(min(len(data),len(w))/2)
                    _r = min(len(data),len(w))-_l-1
            #         _cwt[:,n] = np.abs(np.correlate(data, w, mode='same')) 
                    _cwt[wavelet][_l:-_r,n] = (0.5*convolve(data, w, mode='valid')) 
                    _cwt[wavelet][:,n] += np.abs(_cwt[wavelet][:,n])
        else:
            _cwt[wavelet] = np.zeros((len(data),len(wvlts[wavelet]['w'])))
            if np.iscomplexobj(wvlts[wavelet]['w'][0]):
                for n, w in enumerate(wvlts[wavelet]['w']):
                    _l = floor(min(len(data),len(w))/2)
                    _r = min(len(data),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = np.abs(convolve(data, w, mode='valid')) 
        #             _cwt[:,n] = (0.5*np.correlate(data, w, mode='same')) 
        #             _cwt[:,n] += np.abs(_cwt[:,n])      
            else:
                for n, w in enumerate(wvlts[wavelet]['w']):
                    _l = floor(min(len(data),len(w))/2)
                    _r = min(len(data),len(w))-_l-1
            #         _cwt[:,n] = np.abs(np.correlate(data, w, mode='same')) 
                    _cwt[wavelet][_l:-_r,n] = (0.5*convolve(data, w, mode='valid')) 
                    _cwt[wavelet][:,n] += np.abs(_cwt[wavelet][:,n])
    if show_wavelets:
        plt.legend()
        plt.show()
    return _cwt, wvlts
def local_maxima(cwt, wavelets, scales, threshold, macro_clusters=True, use_scratch=True, extent=1):
    all_events = np.empty((0,), dtype=d_type)
    if use_scratch:
        cwt = h5py.File('cwt.scratch', 'r')
    for wavelet,wvlts in wavelets.items():
        for n, w in enumerate(wvlts['w']):
            _index, _ = find_peaks(cwt[wavelet][:,n], distance=wvlts['N']*scales[n], height=threshold)
            all_events = np.append(all_events, np.array(list(zip((_index), [scales[n]]*len(_index), cwt[wavelet][_index,n], [wvlts['N']]*len(_index))), dtype=d_type), axis=0)
    if macro_clusters:
        all_events_t_l = all_events['time']-0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
        _index_l = np.argsort(all_events_t_l) 
        all_events_t_r = all_events['time']+0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
        _index_r = np.argsort(all_events_t_r) 
        all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]] 
        _slices = np.argwhere(all_events_overlap <= 0).flatten()+1 
        _mc = np.split(all_events[_index_l], _slices, axis=0) 
        if use_scratch:
            cwt.close()
        return _mc
    else:
        if use_scratch:
            cwt.close()        
        return [all_events]
def ucluster(args):
    events, selectivity, w, h = args
    selected_events = []
    events = np.sort(events,order='coeff')[::-1]
    while(len(events) > selectivity):
        _n_0, _n_i = events['N'][0], events['N'] 
        _t_0, _t_i = events['time'][0], events['time'] 
        _s_0, _s_i = events['scale'][0], events['scale'] 
        _c_0, _c_i = events['coeff'][0], events['coeff'] 
        _dt = _t_0 - _t_i 
        _ds = _s_0 - _s_i 
        _theta_i = np.arctan(_ds/_dt) 
        _dist_square = _dt**2 + _ds**2 
        _r_0 = (w*h*_n_0*_s_0)/np.sqrt((w*_n_0*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2) 
        _r_i = (w*h*_n_i*_s_i)/np.sqrt((w*_n_i*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2)*np.sqrt((_c_i-_c_i[-1])/(_c_0-_c_i[-1]))
        _dr_square = (_r_i+_r_0)**2
        _adjacency = np.argwhere(np.nan_to_num(_dr_square,np.inf)-_dist_square >= 0).flatten()
        if len(_adjacency) > selectivity: selected_events.append(events[0])
        events = np.delete(events,_adjacency)
    return np.array(selected_events, dtype=d_type)

    
def tprfdr(t,d,e=1,MS=0):
    tp = []
    D = len(d)
    if D == 0: return 0, 0
    for tr in t:
        error = e*(1-MS*(1-tr))
        try:
            if np.min(np.abs(d-tr)) < error:
                tp.append(np.min(d-tr))
                d = np.delete(d, np.abs(np.argmin(d-tr).flatten()[0]))
        except Exception as excpt:
#             print(excpt)
            pass
#     print(f'TPR: {tpr:.3f}\nFDR: {fdr:.3f}')
    tpr = len(tp)/len(t)
    fdr = (D-len(tp))/D
    return tpr, fdr

class PCWA:
    def __init__(self,dt=1e-5,parallel=False,Mcluster=True,logscale=True,wavelet=['ricker'],scales=[0.01e-3,0.1e-3,30],selectivity=0.5,w=2,h=6,trace=None,show_wavelets=False,update_cwt=True,usescratchfile=False):
        self.dt = dt
        self.parallel = parallel
        self.Mcluster = Mcluster
        self.logscale = logscale
        self.wavelet = wavelet
        self.scales = scales
        self.selectivity = selectivity
        self.w, self.h = w, h
        self.trace = trace
        self.events = []
        self.show_wavelets = show_wavelets
        self.update_cwt = update_cwt
        self.usescratchfile = usescratchfile
        self.cwt = {}
        self.wavelets = {}
        
    def detect_events(self,threshold, trace=None, wavelet=None, scales=None):
        if type(trace) != None:
            self.trace = trace
        if wavelet != None:
            self.wavelet = wavelet
        if scales != None:
            self.scales = scales
        if self.logscale:
            _scales = np.logspace(np.log10(self.scales[0]), np.log10(self.scales[1]), self.scales[2], dtype=np.float64)/self.dt
        else:
            _scales = np.linspace(self.scales[0], self.scales[1], self.scales[2], dtype=np.float64)/self.dt
        if type(self.wavelet) != list:
            self.wavelet = [self.wavelet]
        selected_events = []
#         print(self.wavelet)
        if self.parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                if self.update_cwt:
                    self.cwt, self.wavelets = {}, {}
                    self.cwt, self.wavelets = cwt(self.trace, _scales, self.wavelet, show_wavelets=self.show_wavelets)
                clusters = local_maxima(self.cwt,self.wavelets,_scales,threshold,self.Mcluster,1)
                args = ((cluster,int(len(_scales)*self.selectivity),self.w,self.h) for cluster in clusters)
            #         for n,island in enumerate(islands): 
            #             selected_events.append(select_events(island,selectivity,w,h))
                r = pool.map_async(ucluster, args, chunksize=100)
                [selected_events.append(e) for e in r.get()]
                self.events = np.concatenate(tuple(selected_events),axis=0)
        else:
            if self.update_cwt:
                self.cwt, self.wavelets = {}, {}
                self.cwt, self.wavelets = cwt(self.trace, _scales, self.wavelet, show_wavelets=self.show_wavelets, use_scratch=self.usescratchfile)
            clusters = local_maxima(self.cwt,self.wavelets,_scales,threshold,self.Mcluster,self.usescratchfile,1)
            print(len(clusters))
            args = ((cluster,int(len(_scales)*self.selectivity),self.w,self.h) for cluster in clusters)
        #         for n,island in enumerate(islands): 
        #             selected_events.append(select_events(island,selectivity,w,h))
            for e in map(ucluster, args):
                selected_events.append(e)
            self.events = np.concatenate(tuple(selected_events),axis=0)
        return self.events
