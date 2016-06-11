__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

def up_crossings(data, threshold=0):
    pos = data > threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0]

def down_crossings(data, threshold=0):
    pos = data > threshold
    return (pos[:-1] & ~pos[1:]).nonzero()[0]

def all_crossings(data, threshold=0):
    pos = data > threshold
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def thresholdOnset(data, threshold=0, direction='up', fs=10000.):
    '''

    :param data: time trace
    :param threshold: threshold value
    :param direction: 'up', 'down', 'both'
    :param fs: sampling rate
    :return: timing of each crossing
    '''

    if direction == 'up': onsetInd = up_crossings(data, threshold)
    elif direction == 'down': onsetInd = down_crossings(data, threshold)
    elif direction == 'both': onsetInd = all_crossings(data, threshold)
    return onsetInd/float(fs)

def discreteCrossCorrelation(ts1,ts2,range=(-1.,1.),bins=100, isPlot=False):

    binWidth = (float(range[1])-float(range[0]))/bins
    t = np.arange((range[0]+binWidth/2),(range[1]+binWidth/2),binWidth)
    intervals = list(np.array([(t-binWidth/2),(t+binWidth/2)]).transpose())
    values = np.zeros(bins)

    for ts in list(ts1):
        currIntervals = [x + ts for x in intervals]
        for i, interval in enumerate(currIntervals):
            values[i] += len(np.where(np.logical_and(ts2>interval[0],ts2<=interval[1]))[0])

    if isPlot:
        f = plt.figure(figsize=(15,4)); ax = f.add_subplot(111)
        ax.bar([a[0] for a in intervals],values,binWidth*0.9);ax.set_xticks(t)

    return t,values

def findNearest(trace,value):
    '''
    return the index in "trace" having the closest value to "value"
    '''

    return np.argmin(np.abs(trace-value))

def getOnsetTimeStamps(trace, Fs=10000., threshold = 3., onsetType='raising'):
    '''
    param trace: time trace of digital signal recorded as analog
    param Fs: sampling rate
    return onset time stamps
    '''

    pos = trace > threshold
    if onsetType == 'raising':
        return ((~pos[:-1] & pos[1:]).nonzero()[0]+1)/float(Fs)
    if onsetType == 'falling':
        return ((pos[:-1] & ~pos[1:]).nonzero()[0]+1)/float(Fs)

def power_spectrum(trace, fs, is_plot=False):
    '''
    return power spectrum of a signal trace (should be real numbers) at sampling rate of fs
    '''
    spectrum = np.abs(np.fft.rfft(trace))**2
    freqs = np.fft.rfftfreq(trace.size, 1. / fs)

    if is_plot:
        f=plt.figure()
        idx = np.argsort(freqs)
        plt.plot(freqs[idx], spectrum[idx])
        plt.xlabel('frequency (Hz)')
        plt.ylabel('power')
        plt.show()

    return spectrum, freqs

def sliding_power_spectrum(trace, fs, sliding_window_length, sliding_step_length=None, is_plot=False):
    '''
    calculate power_spectrum of a given trace over time

    trace: input signal trace
    fs: sampling rate (Hz)
    sliding_window_length: length of sliding window (sec)
    sliding_step_length: length of sliding step (sec), if None, equal to sliding_window_length
    is_plot: bool, to plot or not

    :return
    spectrum: 2d array, power at each frequency at each time,
              time is from the first column to the last column
              frequence is from the last row to the first row
    times: time stamp for each column (starting point of each sliding window)
    freqs: frequency for each row (from low to high)
    '''

    if len(trace.shape) != 1: raise ValueError, 'Input trace should be 1d array!'

    total_length = len(trace) / float(fs)

    time_line = np.arange(len(trace)) * (1. / fs)

    if sliding_step_length is None: sliding_step_length = sliding_window_length
    if sliding_step_length > sliding_window_length: print "Step length larger than window length, not using all data points!"
    times = np.arange(0., total_length, sliding_step_length)
    times = times[times + sliding_window_length < total_length]

    if len(times) == 0: raise ValueError, 'No time point found.'
    else:
        points_in_window = int(sliding_window_length * fs)
        if points_in_window <= 0: raise ValueError, 'Sliding window length too short!'
        else:
            freqs = np.fft.rfftfreq(points_in_window, 1. / fs)
            sorting_idx = np.argsort(freqs)
            spectrum = np.empty((len(freqs), len(times)))
            for idx, time in enumerate(times):
                starting_point = findNearest(time_line, time)
                ending_point = starting_point + points_in_window
                current_trace = trace[starting_point:ending_point]
                current_spectrum, _ = power_spectrum(current_trace, fs, is_plot=False)
                spectrum[:,idx] = current_spectrum[sorting_idx]

    if is_plot:
        f = plt.figure(); ax = f.add_subplot(111)
        ax.imshow(spectrum,cmap='jet',interpolation='nearest')
        ax.set_xlabel('times (sec)')
        ax.set_ylabel('frequency (Hz)')
        ax.invert_yaxis()
        ax.set_xticks(times[0::10])
        ax.set_yticks(freqs[0::10])
        plt.show()

    return spectrum, times, freqs



if __name__=='__main__':

    #============================================================================================================
    # a=np.arange(100,dtype=np.float)
    # b=a+0.5+(np.random.rand(100)-0.5)*0.1
    # c=discreteCrossCorrelation(a,b,range=(0,1),bins=50,isPlot=True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    # trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    # ts = getOnsetTimeStamps(trace, Fs=10000., onsetType='raising')
    # assert(ts[2] == 0.0025)
    # ts2 = getOnsetTimeStamps(trace, Fs=10000., onsetType='falling')
    # assert(ts2[2] == 0.0030)
    #============================================================================================================

    #============================================================================================================
    # trace = np.random.rand(300) - 0.5
    # _, _ = power_spectrum(trace, 0.1, True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    time_line = np.arange(5000) * 0.01
    trace = np.sin(time_line * (2 * np.pi))
    trace2 = np.cos(np.arange(2500) * 0.05 * (2 * np.pi))
    trace3 = np.cos(np.arange(2500) * 0.1 * (2 * np.pi))
    trace = trace + np.concatenate((trace2, trace3))

    spectrum, times, freqs = sliding_power_spectrum(trace, 100, 1., is_plot=True)
    print 'times:',times
    print 'freqs:', freqs
    #============================================================================================================

    print 'for debugging...'