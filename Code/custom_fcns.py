######### Custom functions 
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import sem
import string
import scipy.signal
from scipy.signal import find_peaks


############# find nearest
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

############# safe div
def safe_div(x,y):
    return 1.0*x/y if y else 0


############# cus atan
def cus_atan(y,x):
    if x >= 0 and y >= 0:
    	return math.atan2(abs(y), abs(x)) + 1*np.pi/2
    #
    if x >= 0 and y < 0 :
    	return -math.atan2(abs(y), abs(x)) + 1*np.pi/2
    #
    if x < 0 and y < 0 :
    	return math.atan2(abs(y), abs(x)) - 1*np.pi/2
    #
    if x < 0 and y > 0 :
    	return -math.atan2(abs(y), abs(x)) - 1*np.pi/2


############# rev finder
def rev_finder(t, theta):
    he = 7*2*np.pi
    th = 10*np.pi/180
    di = 60
    #
    he = None
    th = None
    #di = None
    pr = 10*np.pi/180
    #
    peaks_p, properties = find_peaks(theta, height=he, threshold=th, distance=di, prominence=pr)
    peaks_n, properties = find_peaks(-theta, height=he, threshold=th, distance=di, prominence=pr)
    #
    theta_p = theta[peaks_p]
    theta_n = theta[peaks_n]
    #
    t_p = t[peaks_p]
    t_n = t[peaks_n]
    #
    theta_peaks = np.hstack([theta_p, theta_n])
    t_peaks = np.hstack([t_p, t_n])
    #
    peaks = np.zeros((len(theta_peaks),2))
    for ii in range(len(theta_peaks)):
        peaks[ii,0] = t_peaks[ii]
        peaks[ii,1] = theta_peaks[ii]
    #
    peaks = peaks[peaks[:,0].argsort()]
    #
    t_peaks = peaks[:,0]
    theta_peaks = peaks[:,1]
    #
    return t_peaks, theta_peaks
    

#################### confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m+h, m-h


########### von Mises
def vonmises_kde(data, kappa, n_bins=100):
    from scipy.special import i0
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde
    
########################## adjust axis
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 5 points
            spine.set_position(('outward', 15))  # outward by 5 points
            spine.set_smart_bounds(True)
            spine.set_linewidth(0.5)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', length=3, width=0.5, colors='k',grid_color='k', grid_alpha=1)
        #ax.tick_params(direction='in', length=2, width=1, colors='k',grid_color='k', grid_alpha=1)
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([]) 