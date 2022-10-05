import numpy as np
import pandas as pd
import h5py
from numpy.lib.arraysetops import isin
from scipy.special import erf
from scipy.special import erf
from scipy.signal import find_peaks, convolve
from math import floor, ceil
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

d_type = np.dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8'), ('N', 'f8'),('class', 'u1')])
eps = 1.0e-12

#================================= Wavelet functions =================================#
def ricker(scale=10, window=1, dx=1):
    """ Creates a scaled Ricker (Mexican Hat) wavelet
    Parameters
    ----------
    scale : float
        The scale applied on the mother wavelet, here it is equivalent to FWHM (default is 10).
    Window : float
        The length of wavelet, can extend or crop the wavelet function on both sides (default is 1).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    wavelet : ndarray of floats
        Array of scaled Ricker wavelet function.

    """
    resolution = scale/dx
    length = int((10*window)*resolution)
    a = resolution/1.25187536
    t = np.arange(length)
    s = 2/(np.sqrt(3*a)*np.pi**1/4)*(1-(t-length/2)**2/a**2)\
        *np.exp(-(t-length/2)**2/(2*a**2))
    s_square_norm = np.trapz(s**2, dx=1)
    s -= np.mean(s)
    return s/np.sqrt(s_square_norm)

def morlet(scale=10, N=6, window=1, is_complex=False, dx=1):
    """ Creates a scaled Morlet wavelet
    Parameters
    ----------
    scale : float
        The scale applied on the mother wavelet, here it is equivalent to FWHM (default is 10).
    N : int
        The number of peaks (effective) (default is 6)
    Window : float
        The length of wavelet, can extend or crop the wavelet function on both sides (default is 1).
    is_complex : bool
        True value will use complex Morlet wavelet (default is False).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    wavelet : ndarray of floats
        Array of scaled Morlet wavelet function.
    """
    resolution = scale/dx
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
    """ Creates a skewed normal function
    Parameters
    ----------
    scale : float
        The scale applied on the mother wavelet, here it is equivalent to FWHM (default is 10).
    N : int
        The number of peaks (effective) (default is 6)
    Window : float
        The length of wavelet, can extend or crop the wavelet function on both sides (default is 1).
    is_complex : bool
        True value will use complex Morlet wavelet (default is False).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    skew_normal : ndarray of floats
        Array of skewed normal function.
    """
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

def msg(scale=10, N=6, window=1, mplx_ratio=[1], mod=0.5, shift=1, skewness=1, is_complex=False, is_mf=False, dx=1):
    """ Creates a combinatorial (multiplexed) Multi-spot Gaussian with N peaks (MSG-N) wavelet function.
     useful to detect events of multiple peaks.
    Parameters
    ----------
    scale : float
        The scale applied on the mother wavelet, here it is equivalent to FWHM (default is 10).
    N : int
        The number of peaks (effective) (default is 6)
    Window : float
        The length of wavelet, can extend or crop the wavelet function on both sides (default is 1).
    mplx_ratio : list of floats, optional
        The ratio of MSG signal multiplexing (weigth of each MSG signal) (default is [1]).
    mod : float
        Modulation ratio of peak width vs the gap between subsequent peaks (default is 0.5).
    shift : float
        The shift of side negative peaks from the last positive peaks on each end (default is 1).
    skewness : float
        The skewness applied for the side negative peaks. 
        Helps with tuning the sensitivity of differntiating between MSG-N vs MSG-(N+1) events (default is 1).
    is_complex : bool
        True value will use complex Morlet wavelet (default is False).
    is_mf : bool
        If True, wavelet is used as a matched filter (doesn't have the side negative peaks) (default is False).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    wavelet : ndarray of floats
        Array of MSG-N wavelet function.
    """
    N = int(N)
    if (type(N) != list):
        N = [N]
    if ((type(mplx_ratio) == float) or (type(mplx_ratio) == int)):
        mplx_ratio = [mplx_ratio]*len(N)
    elif (len(mplx_ratio) != len(N)):
        raise ValueError('multiplex ratio should have same length as N')
    N_max = max(N)
    skewness *= 1.2*mod
    resolution = scale/dx
    sigma = resolution/4.29193*mod*N_max/8
    length = int(N_max*resolution + shift*resolution + (1+4*skewness)*5*N_max*window*sigma)
    t = np.arange(length)
    if is_complex:
        s = np.zeros((length,),dtype=np.complex)
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*skew_normal(t,length/2+((n-1)/2-m+0.125)*(N_max/n)*resolution,sigma, alpha=0)
                s += 1j*mplx_ratio[i]*skew_normal(t,length/2+((n-1)/2-m-0.125)*(N_max/n)*resolution,sigma, alpha=0)
        s -= 0.5*skew_normal(t, length/2+(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 0.5j*skew_normal(t, length/2+(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 0.5*skew_normal(t, length/2-(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= 0.5j*skew_normal(t, length/2-(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
    else:
        s = np.zeros((length,))
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*skew_normal(t,length/2-((n-1)/2-m)*(N_max/n)*resolution,sigma, alpha=0)
    if not is_mf:
        s -= 0.5*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 0.5*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
    return s

def msg_encoded(scale=10, pattern='F08', window=1, mod=0.5, shift=1, skewness=1, is_complex=False, is_mf=False, dx=1):
    """ Creates an encoded Multi-spot Gaussian with N peaks (MSGE-N) wavelet function.

    useful to detect events of multiple encoded peaks to perfectly match multi-spot signal pattern.
    Parameters
    ----------
    scale : float
        The scale applied on the mother wavelet, here it is equivalent to FWHM (default is 10).
    pattern : str
        The string of Hex values determining weight of each peak. 
        i.e. a value of 'F08' means 3 placement of peaks, first 100% height, 
        second 0 (skipped peak), and the third 50% height (default is 'F08').
    Window : float
        The length of wavelet, can extend or crop the wavelet function on both sides (default is 1).
    mod : float
        Modulation ratio of peak width vs the gap between subsequent peaks (default is 0.5).
    shift : float
        The shift of side negative peaks from the last positive peaks on each end (default is 1).
    skewness : float
        The skewness applied for the side negative peaks. 
        Helps with tuning the sensitivity of differntiating between MSG-N vs MSG-(N+1) events (default is 1).
    is_complex : bool
        True value will use complex Morlet wavelet (default is False).
    is_mf : bool
        If True, wavelet is used as a matched filter (doesn't have the side negative peaks) (default is False).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    wavelet : ndarray of floats
        Array of MSGE-N wavelet function.
    """
    if (type(pattern) != list):
        pattern = [pattern]
    N = [0]*len(pattern)
    for i,n in enumerate(pattern):
        pattern[i] = [float.fromhex('0x'+p) for p in pattern[i]]
        N[i] = len(pattern[i])
    N_max = max(N)
    skewness *= 1.2*mod
    resolution = scale/dx
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
    if not is_mf:
        s -= np.sum(pattern)/2*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(pattern)/2*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
    return s

#============================== CWT and Event Detection ==============================#
def cwt(trace, scales, wavelets, use_scratch=True, show_wavelets=False):
    """ Calculates CWT coefficients for the input trace based on the list of 
    wavelet functions and scales provided.
    Parameters
    ----------
    trace : array_like
        Array to calculate CWT coefficients.
    scales : array_like
        Array of scale values.
    wavelets : list of str
        List of wavelet function names available in 
        PCWA ('ricker','morlet-N', 'cmorlet-N', 'msg-N', 'msge-#') 
        where N is the number of peaks for multi-peak signals.
        # for msge wavelet is the encoding string. i.e. msge-F08
    use_scratch : bool
        Use a scratch file on disk (hdf5 format) for CWT coefficients. 
        Set True if you need to store CWT coefficient for later uses.
    show_wavelets : bool
        Plots a sample of wavelet functions for confirmation purposes.
    skewness : float
        The skewness applied for the side negative peaks. 
        Helps with tuning the sensitivity of differntiating between MSG-N vs MSG-(N+1) events (default is 1).
    is_complex : bool
        True value will use complex Morlet wavelet (default is False).
    is_mf : bool
        If True, wavelet is used as a matched filter (doesn't have the side negative peaks) (default is False).
    dx : float
        The sampling step size (default is 1).
    Returns
    -------
    cwt : dict
        Dictionary of CWT coefficients matrices calculated for each wavelet.
    wavelete : dict
        Dictionary of generated wavelet functions at provided scales.
    """
    wvlts = {'wavelets':{wavelet:{} for wavelet in wavelets}}
    wvlts['scales'] = scales
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
    for k,wavelet in enumerate(wavelets):
        if wavelet.lower() == 'ricker': 
            N = 1 
            wvlts['wavelets'][wavelet] = {'N':N, 'w':[ricker(s) for s in scales]}
        elif wavelet[:4].lower() == 'msg-':
            N = int(wavelet[4:]) 
            wvlts['wavelets'][wavelet] = {'N':N, 'w':[msg(s, N=N, mod=0.8, shift=1, skewness=0.5) for s in scales]} 
        elif wavelet[:5].lower() == 'msge-':
            N = len(wavelet[5:]) 
            wvlts['wavelets'][wavelet] = {'N':N, 'w':[msg_encoded(s, pattern=wavelet[5:], mod=1.5, shift=-2.9, skewness=0.04) for s in scales]}
        elif wavelet[:7].lower() == 'morlet-':
            N = int(wavelet[7:]) 
            wvlts['wavelets'][wavelet] = {'N':N, 'w':[morlet(s, N=N, is_complex=False) for s in scales]}
        elif wavelet[:8].lower() == 'cmorlet-':
            N = int(wavelet[8:])
            wvlts['wavelets'][wavelet] = {'N':N, 'w':[morlet(s, N=N, is_complex=True) for s in scales]}
        else:
            raise ValueError('please use proper wavelet names.')
        if show_wavelets:
            plt.plot(wvlts[wavelet]['w'][0],label=wavelet)
        if len(trace) <= len(wvlts['wavelets'][wavelet]['w'][-1]):
            raise RuntimeError('Wavelets are longer than trace, shrink the scale range.')
        if use_scratch:
            _cwt.create_dataset(wavelet, (len(trace),len(wvlts['wavelets'][wavelet]['w'])), chunks=True, dtype='float', compression="gzip")
            if np.iscomplexobj(wvlts['wavelets'][wavelet]['w'][0]):
                for n, w in enumerate(wvlts['wavelets'][wavelet]['w']):
                    _l = floor(min(len(trace),len(w))/2)
                    _r = min(len(trace),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = np.abs(convolve(trace, w, mode='valid')) 
            else:
                for n, w in enumerate(wvlts['wavelets'][wavelet]['w']):
                    _l = floor(min(len(trace),len(w))/2)
                    _r = min(len(trace),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = (0.5*convolve(trace, w, mode='valid')) 
                    _cwt[wavelet][:,n] += np.abs(_cwt[wavelet][:,n])
        else:
            _cwt[wavelet] = np.zeros((len(trace),len(wvlts['wavelets'][wavelet]['w'])))
            if np.iscomplexobj(wvlts['wavelets'][wavelet]['w'][0]):
                for n, w in enumerate(wvlts['wavelets'][wavelet]['w']):
                    _l = floor(min(len(trace),len(w))/2)
                    _r = min(len(trace),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = np.abs(convolve(trace, w, mode='valid')) 
            else:
                for n, w in enumerate(wvlts['wavelets'][wavelet]['w']):
                    _l = floor(min(len(trace),len(w))/2)
                    _r = min(len(trace),len(w))-_l-1
                    _cwt[wavelet][_l:-_r,n] = (0.5*convolve(trace, w, mode='valid')) 
                    _cwt[wavelet][:,n] += np.abs(_cwt[wavelet][:,n])
    if show_wavelets:
        plt.legend()
        plt.show()
    return _cwt, wvlts
def local_maxima(cwt, wavelets, events_scales, threshold, macro_clusters=True, use_scratch=True, extent=1):
    """ Finds and extract local maxima at each scale from CWT coefficients. 
    This is used for the outputs of the cwt() function.
    Parameters
    ----------
    cwt : dict
        Dictionary of CWT coefficients using cwt() function.
    wavelets : dict
        Dictionary of generated wavelets using cwt() function.
    threshold : float
        Threshold for the local maxima detection. It is a value in CWT coefficients domain.
    macro_cluster : bool
        Use macro clustering step to split original candidate events (local maxima points) into 
        macro-clusters. Useful to speed-up in the next step of event filteration (micro-clustering).
        (default is True)
    use_scratch : bool
        Read CWT coefficients from the scratch file on disk (hdf5 format). File name must be 'cwt.scratch'.
        (default is True).
    extent : float
        Extension/spreading of events merging distance in x/time axis for macro-cluster forming purposes (default is 1).
    Returns
    -------
    local maxima : list of ndarray of dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8'), ('N', 'f8'), ('class', 'u1')])
        List of array(s) of local maxima events.
    """
    all_events = np.empty((0,), dtype=d_type)
    if use_scratch:
        cwt = h5py.File('cwt.scratch', 'r')
    k = 0
    for wavelet,wvlts in wavelets['wavelets'].items():
        # print(wvlts)
        for n, w in enumerate(wvlts['w']):
            if events_scales[n]:
                _index, _ = find_peaks(cwt[wavelet][:,n], distance=wvlts['N']*wavelets['scales'][n], height=threshold)
                all_events = np.append(all_events, np.array(list(zip((_index), [wavelets['scales'][n]]*len(_index), cwt[wavelet][_index,n], [0]*len(_index), [wvlts['N']]*len(_index), [k]*len(_index))), dtype=d_type), axis=0)
        k =+ 1
    if macro_clusters:
        all_events_t_l = all_events['loc']-0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
        _index_l = np.argsort(all_events_t_l) 
        all_events_t_r = all_events['loc']+0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
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
def cwt_local_maxima(trace,scales,events_scales,wavelets,threshold,macro_clusters=True,show_wavelets=False,extent=1):
    """ Finds and extract local maxima at each scale while calculating CWT coefficients. 
    This is the integrated version of cwt() and local_maxima() function applied on input signal.
    Parameters
    ----------
    trace : array_like
        Array to calculate CWT coefficients.
    scales : array_like
        Array of scale values.
    wavelets : list of str
        List of wavelet function names available in 
        PCWA ('ricker','morlet-N', 'cmorlet-N', 'msg-N', 'msge-#') 
        where N is the number of peaks for multi-peak signals.
        # for msge wavelet is the encoding string. i.e. msge-F08
    threshold : float
        Threshold for the local maxima detection. It is a value in CWT coefficients domain.
    macro_cluster : bool
        Use macro clustering step to split original candidate events (local maxima points) into 
        macro-clusters. Useful to speed-up in the next step of event filteration (micro-clustering).
        (default is True)
    show_wavelets : bool
        Plots a sample of wavelet functions for confirmation purposes (default is False).
    extent : float
        Extension/spreading of events merging distance in x/time axis for macro-cluster forming purposes (default is 1).
    Returns
    -------
    local maxima : list of ndarray of dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8') ('N', 'f8'), ('class', 'u1')])
        List of array(s) of local maxima events.
    """
    all_events = np.empty((0,), dtype=d_type)
    wvlts = {wavelet:{} for wavelet in wavelets}
    if show_wavelets:
        plt.figure()
    k = 0
    for wavelet in wavelets:
        if wavelet == 'ricker': 
            N = 1 
            wvlts[wavelet] = {'N':N, 'w':[ricker(s) for s in scales]}
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
            wvlts[wavelet] = {'N':N, 'w':[morlet(s, N=N, is_complex=True) for s in scales]}
        if show_wavelets:
            plt.plot(wvlts[wavelet]['w'][0],label=wavelet)
        if np.iscomplexobj(wvlts[wavelet]['w'][0]):
            for n, w in enumerate(wvlts[wavelet]['w']):
                _l = floor(min(len(trace),len(w))/2)
                _cwt = np.abs(convolve(trace, w, mode='valid'))
                if events_scales[n]:
                    _index, _ = find_peaks(_cwt, distance=wvlts[wavelet]['N']*scales[n], height=threshold)
                    all_events = np.append(all_events, np.array(list(zip((_index+_l), [scales[n]]*len(_index), _cwt[_index], [0]*len(_index), [wvlts[wavelet]['N']]*len(_index), [k]*len(_index))), dtype=d_type), axis=0)
        else:
            for n, w in enumerate(wvlts[wavelet]['w']):
                _l = floor(min(len(trace),len(w))/2)
                _cwt = (0.5*convolve(trace, w, mode='valid')) 
                _cwt += np.abs(_cwt)
                if events_scales[n]:
                    _index, _ = find_peaks(_cwt, distance=wvlts[wavelet]['N']*scales[n], height=threshold)
                    all_events = np.append(all_events, np.array(list(zip((_index+_l), [scales[n]]*len(_index), _cwt[_index], [0]*len(_index), [wvlts[wavelet]['N']]*len(_index), [k]*len(_index))), dtype=d_type), axis=0)
        k += 1
    if show_wavelets:
        plt.legend()
        plt.show()
    if macro_clusters:
        all_events_t_l = all_events['loc']-0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
        _index_l = np.argsort(all_events_t_l) 
        all_events_t_r = all_events['loc']+0.5*extent*np.multiply(all_events['N'],all_events['scale']) 
        _index_r = np.argsort(all_events_t_r) 
        all_events_overlap = all_events_t_r[_index_r[:-1]]-all_events_t_l[_index_l[1:]] 
        _slices = np.argwhere(all_events_overlap <= 0).flatten()+1 
        _mc = np.split(all_events[_index_l], _slices, axis=0) 
        return _mc
    else:
        return [all_events]
def ucluster(events, selectivity, w, h):
    """ Finds and extract micro-clusters from the the array of local-maxima (events).
    It forms clusters by finding nearby events using euclidean distance, coefficient value of centroid point,
    and spreading weights along x/y-axis.
    Parameters
    ----------
    events : array_like
        Array to events (local maxima).
    selectivity : float
        Minimum number of events required to form a micro-cluster.
    w : float
        Spreading weight in x (time) axis.
    h : float
        Spreading weight in y (scale) axis.
    Returns
    -------
    events : array_like of dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8'), ('N', 'f8'), ('class', 'u1')])
        Array of selected event(s) found in the given input events list (macro-cluster).
    """
    selected_events = []
    events = np.sort(events,order='coeff')[::-1]
    while(len(events) > selectivity):
        _n_0, _n_i = events['N'][0], events['N'] 
        _t_0, _t_i = events['loc'][0], events['loc'] 
        _s_0, _s_i = events['scale'][0], events['scale'] 
        _c_0, _c_i = events['coeff'][0], events['coeff'] 
        _dt = _t_0 - _t_i 
        _ds = _s_0 - _s_i 
        _theta_i = np.arctan(_ds/(_dt+eps)) 
        _dist_square = _dt**2 + _ds**2 
        _r_0 = (w*h*_n_0*_s_0)/np.sqrt((w*_n_0*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2) 
        _r_i = (w*h*_n_i*_s_i)/np.sqrt((w*_n_i*np.sin(_theta_i))**2+(h*np.cos(_theta_i))**2)*np.sqrt((_c_i-_c_i[-1])/(_c_0-_c_i[-1]))
        _dr_square = (_r_i+_r_0)**2
        _adjacency = np.argwhere(np.nan_to_num(_dr_square,np.inf)-_dist_square >= 0).flatten()
        if len(_adjacency) > selectivity: selected_events.append(events[0])
        events = np.delete(events,_adjacency)
    return np.array(selected_events, dtype=d_type)

def ucluster_map(args):
    """ A wrapper function to use ucluster with map().
    Parameters
    ----------
    args : iterable
        Iterable list/dict of arguments
    Returns
    -------
    events : array_like of dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8'), ('N', 'f8'), ('class', 'u1')])
        Array of selected event(s) found in the given input events list (macro-cluster).
    """
    return ucluster(*args)
    
def tprfdr(t,d,e=1,MS=False):
    """
    Calculate TPR and FDR values for general/mass-spectroscopy detected events.
    Parameters
    ----------
    t : array_like
        Ground truth location
    d : array_like
        Detected location
    e : float
        Acceptable error range (tolerance) to consider an event as true event. 
        if MS=0, e value is abslute
        if MS=1, e is relative to location
        (deafult is 1)
    MS: bool
        Set true if used for mass spectroscopy trace (default is False).
    Returns
    -------
    tpr : float
        TPR value
    fdr : float
        FDR value
    """
    tp = []
    D = len(d)
    if MS: MS = 1
    else: MS = 0
    if D == 0: return 0, 0
    for tr in t:
        error = e*(1-MS*(1-tr))
        try:
            if np.min(np.abs(d-tr)) < error:
                tp.append(np.min(d-tr))
                d = np.delete(d, np.abs(np.argmin(d-tr).flatten()[0]))
        except Exception as excpt:
            print(excpt)
            pass
    tpr = len(tp)/len(t)
    fdr = (D-len(tp))/D
    return tpr, fdr

class PCWA:
    def __init__(self,dx=1,parallel=False,mcluster=True,logscale=True,wavelet=['ricker'],scales=[10,100,30],selectivity=0.3,w=2,h=6,extent=1,trace=None,show_wavelets=False,update_cwt=True,keep_cwt=False,use_scratchfile=False):
        self.dx = dx
        self.parallel = parallel
        self.mcluster = mcluster
        self.logscale = logscale
        self.wavelet = wavelet
        self.scales_range = scales
        self.scales_arr = []
        self.events_scales_range = [self.scales_range[0], self.scales_range[1]]
        self.events_scales_arr = []
        self.selectivity = selectivity
        self.w, self.h = w, h
        self.extent = extent
        self.trace = trace
        self.events = []
        self.show_wavelets = show_wavelets
        self.update_cwt = update_cwt
        self.use_scratchfile = use_scratchfile
        self.keep_cwt = keep_cwt
        self.cwt = {}
        self.wavelets = {}
        
    def detect_events(self,threshold, trace=None, wavelet=None, scales=None):
        """
        Detect events using the PCWA algorithm.
        Parameters
        ----------
        threshold : float
            threshold (height) used to find initial local maxima.
        trace : array_like, None
            Input signal to detect events. It will overwrite the class value defined for PCWA.trace. (default is None).
        wavelet : list of strings
            Name of the wavelet function(s) to calculate CWT coefficients. Currently acceptable values are:
            'ricker', 'msg-N', msge-w0w1w2...', 'morlet-N', 'cmorlet-N'
            N: is the number of peaks in a multi-peak wavelet, i.e. msg-8 contains 8 peaks. 
            msge is encoded version of msg where the weight of individual peaks are given as a series of hex numbers. 
            i.e. msge-F8A2 is a msg wavelet with first peak height being 8 times the last peak.
        scales: list
            [minimum, maximum, count] of scales. Scale values are in dx scale. Will overwrite pcwa.scales_arr parameter.
        Returns
        -------
        events : ndarray with dtype([('loc', 'i8'), ('scale', 'f8'), ('coeff', 'f8'), ('height', 'f8'), ('N', 'f8'), ('class', 'u1')])
            Array including information of detected events.
        """
        if type(trace) in [list, np.ndarray, pd.Series]:
            self.trace = np.array(trace).flatten()
        elif type(self.trace) in [list, np.ndarray, pd.Series]:
            trace = np.array(self.trace).flatten()
        else:
            raise RuntimeError('Input trace not valid.')
        if wavelet != None:
            self.wavelet = wavelet
        if scales != None:
            self.scales_range = scales
            if self.logscale:
                self.scales_arr = np.logspace(np.log10(self.scales_range[0]), np.log10(self.scales_range[1]), int(self.scales_range[2]), dtype=np.float64)/self.dx
            else:
                self.scales_arr = np.linspace(self.scales_range[0], self.scales_range[1], int(self.scales_range[2]), dtype=np.float64)/self.dx
        if len(self.scales_arr) == 0:
            if self.scales_range == None:
                raise RuntimeError('Please set pcwa.scales_arr or provide proper scales_range.')
            else:
                if self.logscale:
                    self.scales_arr = np.logspace(np.log10(self.scales_range[0]), np.log10(self.scales_range[1]), int(self.scales_range[2]), dtype=np.float64)/self.dx
                else:
                    self.scales_arr = np.linspace(self.scales_range[0], self.scales_range[1], int(self.scales_range[2]), dtype=np.float64)/self.dx
        if len(self.trace) <= self.scales_arr[-1]:
            raise RuntimeError('Scales are longer than trace, shrink the scale range.')
        if type(self.wavelet) != list:
            self.wavelet = [self.wavelet]
        self.events_scales_arr = (self.scales_arr>=(self.events_scales_range[0]/self.dx))*(self.scales_arr<=(self.events_scales_range[1]/self.dx))
        selected_events = []
        if self.parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                if self.update_cwt:
                    self.cwt, self.wavelets = {}, {}
                    self.cwt, self.wavelets = cwt(self.trace, self.scales_arr, self.wavelet, show_wavelets=self.show_wavelets)
                clusters = local_maxima(self.cwt,self.wavelets,self.events_scales_arr,threshold,self.mcluster,self.extent)
                args = ((cluster,int(len(self.scales_arr)*self.selectivity),self.w,self.h) for cluster in clusters)
                r = pool.map_async(ucluster, args, chunksize=100)
                [selected_events.append(e) for e in r.get()]
                self.events = np.concatenate(tuple(selected_events),axis=0)
        else:
            if self.keep_cwt:
                if self.update_cwt:
                    self.cwt, self.wavelets = {}, {}
                    self.cwt, self.wavelets = cwt(self.trace, self.scales_arr, self.wavelet, show_wavelets=self.show_wavelets, use_scratch=self.use_scratchfile)
                clusters = local_maxima(self.cwt,self.wavelets,self.events_scales_arr,threshold,self.mcluster,self.use_scratchfile,self.extent)
            else:
                clusters = cwt_local_maxima(self.trace,self.scales_arr,self.events_scales_arr,self.wavelet,threshold,self.mcluster,self.show_wavelets,self.extent)
            args = [(cluster,int(len(self.scales_arr)*self.selectivity),self.w,self.h) for cluster in clusters]
            for e in map(ucluster_map, args):
                selected_events.append(e)
            self.events = np.concatenate(tuple(selected_events),axis=0)
        _idx_max = len(self.trace)
        _idx = zip(np.clip(self.events['loc']-0.5*self.events['N']*self.events['scale'],a_min=0,a_max=_idx_max-1).astype(int),\
            np.clip(self.events['loc']+0.5*self.events['N']*self.events['scale'],a_min=0,a_max=_idx_max).astype(int))
        self.events['height'] = np.array([max(self.trace[_i[0]:_i[1]]) for _i in _idx])
        return self.events
    
    def view_events(self,events,span=1,ax=None):
        if type(events) == list:
            if len(events) == 0:
                print("events list cannot be empty")
                return
            N = len(events)
        elif type(events) == int:
            N = 1
            events = [events]
        else:
            print("Wrong value for events. Should be a list of events # or a single integer for event #")
            return
        if ax == None:
            _r,_c = 1+((N-1)//5),min(N,5)
            fig,ax = plt.subplots(_r,_c,figsize=(3*_c,2*_r))
        if N == 1:
            ax = [ax]
        else:
            ax = ax.flatten()
        for n,e in enumerate(events):
            _win = int(span*self.events[e]['N']*self.events[e]['scale'])
            _t = np.arange(int(self.events[e]['loc']-_win),int(self.events[e]['loc']+_win))*self.dx
            ax[n].plot(_t,self.trace[int(self.events[e]['loc']-_win):int(self.events[e]['loc']+_win)],color=f"C{self.events[e]['class']}")
            ax[n].set_xlabel(f"event #{e}")
        plt.show()
        return fig,ax
