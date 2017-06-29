# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:12:24 2014

@author: junz
"""


import os
import datetime
import random
from psychopy import visual, event
import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle

import socket
import tifffile as tf
import core.FileTools as ft
import core.ImageAnalysis as ia


#from zro import RemoteObject, Proxy

try: import IO.nidaq as iodaq
except ImportError as e:
    print e
    print 'import iodaq from aibs package...'
    try: import aibs.iodaq as iodaq
    except ImportError as er: print er



def gaussian(x, mu=0, sig=1.):
    
    return np.exp(np.divide(-np.power(x - mu, 2.) , 2 * np.power(sig, 2.)))


def analyze_frames(ts, refreshRate, checkPoint=(0.02, 0.033, 0.05, 0.1)):
    """
    analyze frame durations. input is the time stamps of each frame and
    the refresh rate of the monitor
    """
    
    frameDuration = ts[1::] - ts[0:-1]
    plt.figure()
    plt.hist(frameDuration, bins=np.linspace(0.0, 0.05, num=51))
    refreshRate = float(refreshRate)
    
    frame_stats = '\n'
    frame_stats += 'Total frame number: %d. \n' % (len(ts)-1)
    frame_stats += 'Total length of display   : %.5f second. \n' % (ts[-1]-ts[0])
    frame_stats += 'Expected length of display: %.5f second. \n' % ((len(ts)-1)/refreshRate)
    frame_stats += 'Mean of frame durations: %.2f ms. \n' % (np.mean(frameDuration)*1000)
    frame_stats += 'Standard deviation of frame durations: %.2f ms. \n' % (np.std(frameDuration)*1000)
    frame_stats += 'Shortest frame: %.2f ms, index: %d. \n' % (min(frameDuration)*1000, np.nonzero(frameDuration==np.min(frameDuration))[0][0])
    frame_stats += 'longest frame : %.2f ms, index: %d. \n' % (max(frameDuration)*1000, np.nonzero(frameDuration==np.max(frameDuration))[0][0])
    
    for i in range(len(checkPoint)):
        checkNumber = checkPoint[i]
        frameNumber = len(frameDuration[frameDuration>checkNumber])
        frame_stats += 'Number of frames longer than %d ms: %d; %.2f%% \n' % (round(checkNumber*1000), frameNumber, round(frameNumber*10000/(len(ts)-1))/100)
    
    print frame_stats
    
    return frameDuration, frame_stats


def noise_movie(frameFilter, widthFilter, heightFilter, isplot = False):
    """
    creating a numpy array with shape [len(frameFilter), len(heightFilter), len(widthFilter)]
    
    this array is random noize filtered by these three filters in Fourier domain
    each pixel of the movie have the value in [-1 1]
    """
    
    rawMov = np.random.rand(len(frameFilter), len(heightFilter), len(widthFilter))
    
    rawMovFFT = np.fft.fftn(rawMov)
    
    filterX = np.repeat(np.array([widthFilter]), len(heightFilter), axis = 0)
    filterY = np.repeat(np.transpose(np.array([heightFilter])), len(widthFilter), axis = 1)
    
    filterXY = filterX * filterY
    
    for i in xrange(rawMovFFT.shape[0]):
        rawMovFFT[i] = frameFilter[i]* (rawMovFFT[i] * filterXY) 
    
    
#    heightFilter = heightFilter.reshape((len(heightFilter),1))
#    frameFilter = frameFilter.reshape((len(frameFilter),1,1))
#    
#    rawMovFFT = np.multiply(np.multiply(np.multiply(rawMovFFT,widthFilter),heightFilter),frameFilter)
    
    filteredMov = np.real(np.fft.ifftn(rawMovFFT))
    
    rangeFilteredMov = np.amax(filteredMov) - np.amin(filteredMov)    
    noise_movie = ((filteredMov - np.amin(filteredMov)) / rangeFilteredMov) * 2 - 1
    
    if isplot:
        tf.imshow(noise_movie, vmin=-1, vmax=1, cmap='gray')
    
    return noise_movie


def generate_filter(length, # length of filter
                   Fs, # sampling frequency
                   Flow, # low cutoff frequency
                   Fhigh, # high cutoff frequency
                   mode = 'box'): # filter mode, '1/f' or 'box'

    """
    generate one dimensional filter on Fourier domain, with symmetrical structure
    """
    
    freqs = np.fft.fftfreq(int(length), d = (1./float(Fs)))
    
    filterArray = np.ones(length)
    
    for i in xrange(len(freqs)):
        if ((freqs[i] > 0) and (freqs[i] < Flow) or (freqs[i] > Fhigh)) or \
           ((freqs[i] < 0) and (freqs[i] > -Flow) or (freqs[i] < -Fhigh)):
            filterArray[i] = 0
    
    if mode == '1/f':
        filterArray[1:] = filterArray[1:] / abs(freqs[1:])
        filterArray[0] = 0
        filterArray = (filterArray - np.amin(filterArray)) / (np.amax(filterArray) - np.amin(filterArray))
    elif mode == 'box':
        filterArray[0] = 0
    else: raise NameError, 'Variable "mode" should be either "1/f" or "box"!'
    
    if Flow == 0:
        filterArray[0] = 1
    
    return filterArray


def lookup_image(img, lookupI, lookupJ):
    """
    generate warpped image from img, using look up talbel: lookupI and lookupJ
    """
    
    if not img.shape == lookupI.shape:
        raise LookupError, 'The image and lookupI should have same size!!'
        
    if not lookupI.shape == lookupJ.shape:
        raise LookupError, 'The lookupI and lookupJ should have same size!!'

    img2 = np.zeros(img.shape)
    
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i,j] = img[lookupI[i,j],lookupJ[i,j]]
            
    return img2


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def get_warped_square(degCorX,degCorY,center,width,height,ori,foregroundColor=1,backgroundColor=0.):
    """
    generate a frame (matrix) with single square defined by center, width, height and orientation in degress
    visual degree value of each pixel is defined by degCorX, and degCorY
    dtype = np.float32, color space, -1:black, 1:white

    ori: angle in degree, should be 0~180
    """

    frame = np.ones(degCorX.shape,dtype=np.float32)*backgroundColor

    if ori < 0. or ori > 180.: raise ValueError, 'ori should be between 0 and 180.'

    k1 = np.tan(ori*np.pi/180.)
    k2 = np.tan((ori+90.)*np.pi/180.)

    disW = np.abs((k1*degCorX - degCorY + center[1] - k1 * center[0]) / np.sqrt(k1**2 +1))
    disH = np.abs((k2*degCorX - degCorY + center[1] - k2 * center[0]) / np.sqrt(k2**2 +1))

    frame[np.logical_and(disW<=width/2.,disH<=height/2.)] = foregroundColor

    return frame


def circle_mask(map_x, map_y, center, radius):
    """
    generate a binary mask of a circle with given center and radius on a map with coordinates for each pixel defined by
    map_x and map_y

    :param map_x: x coordinates for each pixel on a map
    :param map_y: y coordinates for each pixel on a map
    :param center: center coordinates of circle center {x, y}
    :param radius: radius of the circle
    :return: binary mask for the circle, value range [0., 1.]
    """

    if map_x.shape != map_y.shape: raise ValueError, 'map_x and map_y should have same shape!'

    if len(map_x.shape) != 2: raise ValueError, 'map_x and map_y should be 2-d!!'

    circle_mask = np.zeros(map_x.shape, dtype = np.uint8)
    for (i, j), value in  np.ndenumerate(circle_mask):
        x=map_x[i,j]; y=map_y[i,j]
        if ia.distance((x,y),center) <= radius:
            circle_mask[i,j] = 1

    return circle_mask


def get_grating(map_x, map_y, ori=0., spatial_freq=0.1, center=(0.,60.), phase=0., contrast=1.):
    """
    generate a grating frame with defined spatial frequency, center location, phase and contrast

    :param map_x: x coordinates for each pixel on a map
    :param map_y: y coordinates for each pixel on a map
    :param center: center coordinates of circle center {x, y}
    :param spatial_freq: spatial frequency (cycle per unit)
    :param phase: in arc
    :param contrast: [0., 1.]
    :return: a frame as floating point 2-d array with grating, value range [0., 1.]
    """

    if map_x.shape != map_y.shape: raise ValueError, 'map_x and map_y should have same shape!'

    if len(map_x.shape) != 2: raise ValueError, 'map_x and map_y should be 2-d!!'

    map_x_h = np.array(map_x, dtype = np.float32)
    map_y_h = np.array(map_y, dtype = np.float32)

    distance = np.sin(ori) * (map_x_h - center[0]) - np.cos(ori) * (map_y_h - center[1])

    grating = np.sin(distance * 2 * np.pi * spatial_freq + phase)
    grating = (grating + 1.) / 2. # change the scale of grating to be [0., 1.]
    grating = (grating * contrast) + (1 - contrast) / 2 # adjust contrast

    return grating.astype(map_x.dtype)


class Monitor(object):
    """
    monitor object created by Jun, has the method "remap" to generate the 
    spherical corrected coordinates in degrees
    """
    
    def __init__(self, 
                 resolution, 
                 dis, 
                 monWcm, 
                 monHcm, 
                 C2Tcm, 
                 C2Acm, 
                 monTilt, 
                 visualField='right',
                 degCorX=None, 
                 degCorY=None, 
                 name='testMonitor', 
                 gamma=None, 
                 gammaGrid=None, 
                 luminance=None,
                 downSampleRate=10, 
                 refreshRate = 60.):
                     
        if resolution[0] % downSampleRate != 0 or resolution[1] % downSampleRate != 0:           
           raise ArithmeticError, 'Resolution pixel numbers are not divisible by down sampling rate'
        
        self.resolution = resolution
        self.dis = dis
        self.monWcm = monWcm
        self.monHcm = monHcm
        self.C2Tcm = C2Tcm # distance from gaze center to monitor top
        self.C2Acm = C2Acm # distance from gaze center to anterior edge of the monitor
        self.monTilt = monTilt
        self.visualField = visualField
        self.degCorX = degCorX
        self.degCorY = degCorY
        self.name = name
        self.downSampleRate = downSampleRate
        self.gamma = gamma
        self.gammaGrid = gammaGrid
        self.luminance = luminance
        self.refreshRate = 60
        
        #distance form the projection point of the eye to the bottom of the monitor
        self.C2Bcm = self.monHcm - self.C2Tcm
        #distance form the projection point of the eye to the right of the monitor
        self.C2Pcm = self.monWcm - self.C2Acm
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/downSampleRate
        resolution[1]=self.resolution[1]/downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        if self.visualField == "left": 
            mapX = np.linspace(self.C2Acm, -1.0 * self.C2Pcm, resolution[1])
            
        if self.visualField == "right":
            mapX = np.linspace(-1 * self.C2Acm, self.C2Pcm, resolution[1])
            
        mapY = np.linspace(self.C2Tcm, -1.0 * self.C2Bcm, resolution[0])
        oldmapX, oldmapY = np.meshgrid(mapX, mapY, sparse = False)
        
        self.linCorX=oldmapX
        self.linCorY=oldmapY
        
        self.remap()
        
    def set_gamma(self, gamma, gammaGrid):
        self.gamma = gamma
        self.gammaGrid = gammaGrid
        
    def set_luminance(self, luminance):
        self.luminance = luminance
        
    def set_downsample_rate(self, downSampleRate):
        
        if self.resolution[0] % downSampleRate != 0 or self.resolution[1] % downSampleRate != 0:
           
           raise ArithmeticError, 'resolutionolution pixel numbers are not divisible by down sampling rate'
        
        self.downSampleRate=downSampleRate
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/downSampleRate
        resolution[1]=self.resolution[1]/downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        if self.visualField == "left": 
            mapX = np.linspace(self.C2Acm, -1.0 * self.C2Pcm, resolution[1])
            
        if self.visualField == "right":
            mapX = np.linspace(-1 * self.C2Pcm, self.C2Pcm, resolution[1])
            
        mapY = np.linspace(self.C2Tcm, -1.0 * self.C2Bcm, resolution[0])
        oldmapX, oldmapY = np.meshgrid(mapX, mapY, sparse = False)
        
        self.linCorX=oldmapX
        self.linCorY=oldmapY
        
        self.remap()
        
        
    def remap(self):
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/self.downSampleRate
        resolution[1]=self.resolution[1]/self.downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        newmapX = np.zeros(resolution,dtype=np.float16)
        newmapY = np.zeros(resolution,dtype=np.float16)
        
        
        for j in range(resolution[1]):
            newmapX[:, j] = (180.0 / np.pi) * np.arctan(self.linCorX[0, j] / self.dis)
            dis2 = np.sqrt(np.square(self.dis) + np.square(self.linCorX[0, j])) #distance from 
            
            for i in range(resolution[0]):
                newmapY[i, j] = (180.0 / np.pi) * np.arctan(self.linCorY[i, 0] / dis2)
                
        self.degCorX = newmapX+90-self.monTilt
        self.degCorY = newmapY
        
    def plot_map(self):
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/self.downSampleRate
        resolution[1]=self.resolution[1]/self.downSampleRate
        
        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))
        
        f1 = plt.figure(figsize=(12,5))
        f1.suptitle('Remap monitor', fontsize=14, fontweight='bold')
        
        OMX = plt.subplot(221)
        OMX.set_title('Linear Map X (cm)')
        currfig = plt.imshow(self.linCorX)
        levels1 = range(int(np.floor(self.linCorX.min() / 10) * 10), int((np.ceil(self.linCorX.max() / 10)+1) * 10), 10)
        im1 =plt.contour(mapcorX, mapcorY, self.linCorX, levels1, colors = 'k', linewidth = 2)
#        plt.clabel(im1, levels1, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels1)
        plt.gca().set_axis_off()
        
        OMY = plt.subplot(222)
        OMY.set_title('Linear Map Y (cm)')
        currfig = plt.imshow(self.linCorY)
        levels2 = range(int(np.floor(self.linCorY.min() / 10) * 10), int((np.ceil(self.linCorY.max() / 10)+1) * 10), 10)
        im2 =plt.contour(mapcorX, mapcorY, self.linCorY, levels2, colors = 'k', linewidth = 2)
#        plt.clabel(im2, levels2, fontsize = 10, inline = 1, fmt='%2.2f')
        f1.colorbar(currfig,ticks=levels2)
        plt.gca().set_axis_off()
        
        NMX = plt.subplot(223)
        NMX.set_title('Spherical Map X (deg)')
        currfig = plt.imshow(self.degCorX)
        levels3 = range(int(np.floor(self.degCorX.min() / 10) * 10), int((np.ceil(self.degCorX.max() / 10)+1) * 10), 10)
        im3 =plt.contour(mapcorX, mapcorY, self.degCorX, levels3, colors = 'k', linewidth = 2)
#        plt.clabel(im3, levels3, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels3)
        plt.gca().set_axis_off()
        
        NMY = plt.subplot(224)
        NMY.set_title('Spherical Map Y (deg)')
        currfig = plt.imshow(self.degCorY)
        levels4 = range(int(np.floor(self.degCorY.min() / 10) * 10), int((np.ceil(self.degCorY.max() / 10)+1) * 10), 10)
        im4 =plt.contour(mapcorX, mapcorY, self.degCorY, levels4, colors = 'k', linewidth = 2)
#        plt.clabel(im4, levels4, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig,ticks=levels4)
        plt.gca().set_axis_off()
        
    def save_monitor(self):
        pass
    
    def generate_Lookup_table(self):
        """
        generate lookup talbe between degree corrdinates and linear corrdinates
        return two matrix: 
        lookupI: i index in linear matrix to this pixel after warping
        lookupJ: j index in linear matrix to this pixel after warping
        """
        
        #length of one degree on monitor at gaze point
        degDis = np.tan(np.pi / 180) * self.dis
        
        #generate degree coordinate without warpping
        degNoWarpCorX = self.linCorX / degDis
        degNoWarpCorY = self.linCorY / degDis
        
        #deg coordinates
        degCorX = self.degCorX+self.monTilt-90
        degCorY = self.degCorY
        
        lookupI = np.zeros(degCorX.shape).astype(np.int32)
        lookupJ = np.zeros(degCorX.shape).astype(np.int32)
        
        for j in xrange(lookupI.shape[1]):
            currDegX = degCorX[0,j]
            diffDegX = degNoWarpCorX[0,:] - currDegX
            IndJ = np.argmin(np.abs(diffDegX))
            lookupJ[:,j] = IndJ
            
            for i in xrange(lookupI.shape[0]):
                currDegY = degCorY[i,j]
                diffDegY = degNoWarpCorY[:,IndJ] - currDegY
                indI = np.argmin(np.abs(diffDegY))
                lookupI[i,j] = indI
        
        return lookupI, lookupJ


class Indicator(object):
    """
    flashing indicator for photodiode
    """

    def __init__(self,
                 monitor,
                 width_cm = 3.,
                 height_cm = 3.,
                 position = 'northeast',
                 isSync = True,
                 freq = 2.):
        self.monitor=monitor
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.width_pixel, self.height_pixel = self.get_size_pixel()
        self.position = position
        self.centerWpixel, self.centerHpixel = self.get_center()
        self.isSync = isSync

        if isSync == False:
            self.freq = freq #if not synchronized with stimulation, self update frquency of the indicator
            self.frameNum = self.get_frames()
        else:
            self.freq = None
            self.frameNum = None

    def get_size_pixel(self):

        screen_width = self.monitor.resolution[1] / self.monitor.downSampleRate
        screen_height = self.monitor.resolution[0] / self.monitor.downSampleRate

        indicator_width = int((self.width_cm / self.monitor.monWcm ) * screen_width)
        indicator_height = int((self.height_cm / self.monitor.monHcm ) * screen_height)

        return indicator_width, indicator_height

    def get_center(self):

        screen_width = self.monitor.resolution[1] / self.monitor.downSampleRate
        screen_height = self.monitor.resolution[0] / self.monitor.downSampleRate

        if self.position == 'northeast':
            centerW = screen_width - self.width_pixel / 2
            centerH = self.height_pixel / 2

        elif self.position == 'northwest':
            centerW = self.width_pixel / 2
            centerH = self.height_pixel / 2

        elif self.position == 'southeast':
            centerW = screen_width - self.width_pixel / 2
            centerH = screen_height - self.height_pixel / 2

        elif self.position == 'southwest':
            centerW = self.width_pixel / 2
            centerH = screen_height - self.height_pixel / 2

        else:
            raise LookupError, '"position" attributor should be "northeast", "southeast", "northwest" and "southwest"'

        return int(centerW), int(centerH)

    def get_frames(self):

        """
        if not synchronized with stimulation, get frame numbers of each update
        of indicator
        """

        refreshRate = self.monitor.refreshRate

        if refreshRate % self.freq != 0:
            raise ArithmeticError, "self update frequency of should be divisible by monitor's refresh rate."

        return refreshRate/self.freq

        
class Stim(object):
    """
    generic class for visual stimulation
    """
    def __init__(self,
                 monitor, # Monitor object
                 indicator, # indicator object,
                 background = 0., # back ground color [-1,1]
                 coordinate = 'degree', #'degree' or 'linear'
                 preGapDur = 2., # duration of gap period before stimulus, second
                 postGapDur = 3.): # duration of gap period after stimulus, second
        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate
        self.preGapDur = preGapDur
        self.postGapDur = postGapDur

        self.clear()

    @property
    def preGapFrameNum(self):
        return int(self.preGapDur * self.monitor.refreshRate)

    @property
    def postGapFrameNum(self):
        return int(self.postGapDur * self.monitor.refreshRate)

    def generate_frames(self):
        """
        place holder of function "generate_frames" for each specific stimulus
        """
        print 'Nothing executed! This is place holder of function "generate_frames" for each specific stimulus.'
        print 'This function should return a list of tuples, each tuple represents a single frame of the stimulus and contains all the information to recreate the frame.'
        
    def generate_movie(self):
        """
        place holder of function "generate_movie" for each specific stimulus
        """
        print 'Nothing executed! This is place holder of function "generate_movie" for each specific stimulus.'
        print 'This function should return two things:'
        print 'First: a 3-d array (with format of uint8) of the stimulus to be displayed.'
        print 'Second: a dictionary contain the information of this particular stimulus'
        
    def clear(self):
        self.frames = None
    
    def set_pre_gap_dur(self,preGapDur):
        self.preGapFrameNum = int(preGapDur * self.monitor.refreshRate)
        self.clear()
        
    def set_post_gap_dur(self,postGapDur):
        self.postGapFrameNum = int(postGapDur * self.monitor.refreshRate)
        self.clear()


class UniformContrast(Stim):
    """
    full field uniform luminance for recording spontaneous activity.
    """

    def __init__(self, monitor, indicator, duration, color=0., preGapDur=2., postGapDur=3., background=0.,
                 coordinate='degree'):

        super(UniformContrast, self).__init__(monitor=monitor, indicator=indicator,
                                              coordinate=coordinate, background=background,
                                              preGapDur=preGapDur, postGapDur=postGapDur)
        self.stimName = 'UniformContrast'
        self.duration=duration
        self.color=color

    def generate_frames(self):
        """
        generate a tuple of parameters of each frame.

        for each frame:

        first element: gap:0 or display:1
        second element: color of indicator, gap:-1, display:1
        """

        displayFrameNum = int(self.duration * self.monitor.refreshRate)

        frames = [(0, -1)] * self.preGapFrameNum + [(1, 1.)] * displayFrameNum + \
                 [(0, -1)] * self.postGapFrameNum

        return tuple(frames)

    def generate_movie(self):
        """
        generate move for uniform contrast display for recording of spontaneous activity
        """

        self.frames = self.generate_frames()

        fullSequence = np.zeros((len(self.frames), self.monitor.degCorX.shape[0], self.monitor.degCorX.shape[1]),
                                dtype=np.float16)

        indicatorWmin = self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax = self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin = self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax = self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX, 1)),
                                               dtype=np.float16)

        display = self.color * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX, 1)),
                                       dtype=np.float16)

        if not (self.coordinate == 'degree' or self.coordinate == 'linear'):
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'

        for i in range(len(self.frames)):
            currFrame = self.frames[i]

            if currFrame[0] == 0:
                currFCsequence = background
            else:
                currFCsequence = display

            currFCsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[1]

            fullSequence[i] = currFCsequence

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print ['Generating numpy sequence: ' + str(int(100 * (i + 1) / len(self.frames))) + '%']

        mondict = dict(self.monitor.__dict__)
        indicatordict = dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        fullDictionary = {'stimulation': NFdict,
                          'monitor': mondict,
                          'indicator': indicatordict}

        return fullSequence, fullDictionary


class KSstim(Stim):
    """
    generate Kalatsky & Stryker stimulation integrats flashing indicator for 
    photodiode
    """
    def __init__(self,
                 monitor,
                 indicator,
                 background=0., #back ground color [-1,1]
                 coordinate='degree', #'degree' or 'linear'
                 squareSize=25., #size of flickering square
                 squareCenter=(0,0), #coordinate of center point
                 flickerFrame=10,
                 sweepWidth=20., # width of sweeps (unit same as Map, cm or deg)
                 stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                 direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                 sweepFrame=1,
                 iteration=1, 
                 preGapDur=2.,
                 postGapDur=3.):
        
        super(KSstim,self).__init__(monitor=monitor,indicator=indicator,coordinate=coordinate,background=background,preGapDur=preGapDur,postGapDur=postGapDur)
                     
        self.stimName = 'KSstim'
        self.squareSize = squareSize
        self.squareCenter = squareCenter
        self.flickerFrame = flickerFrame
        self.flickerFrequency=self.monitor.refreshRate / self.flickerFrame
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.direction = direction
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.frameConfig = ('isDisplay', 'squarePolarity', 'sweepIndex', 'indicatorColor')
        self.sweepConfig = ('orientation', 'sweepStartCoordinate', 'sweepEndCoordinate')
        
        self.sweepSpeed = self.monitor.refreshRate * self.stepWidth / self.sweepFrame #the speed of sweeps deg/sec
        self.flickerHZ = self.monitor.refreshRate / self.flickerFrame

        self.clear()
        

    def generate_squares(self):
        """
        generate checker board squares
        """
        
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
            
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        negX = np.ceil( abs( ( ( minX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posX = np.ceil( abs( ( ( maxX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        
        negY = np.ceil( abs( ( ( minY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posY = np.ceil( abs( ( ( maxY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        
        squareV = np.ones((np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
        squareV = -1 * squareV
        
        stepV = np.arange(self.squareCenter[0] - ( 2 * negX + 0.5 ) * self.squareSize, 
                          self.squareCenter[0] + ( 2 * posX - 0.5 ) * self.squareSize, self.squareSize*2)
        
        for i in range(len(stepV)):
            squareV[ np.where( np.logical_and( mapX >= stepV[i], mapX < (stepV[i] + self.squareSize)))] = 1.0
        
        squareH = np.ones((np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
        squareH = -1 * squareH
        
        stepH = np.arange(self.squareCenter[1] - ( 2 * negY + 0.5 ) * self.squareSize, 
                          self.squareCenter[1] + ( 2 * posY - 0.5 ) * self.squareSize, self.squareSize*2)
        
        for j in range(len(stepH)):
            squareH[ np.where( np.logical_and( mapY >= stepH[j], mapY < (stepH[j] + self.squareSize)))] = 1
        
        squares = np.multiply(squareV, squareH)
        
        return squares

    def plot_squares(self):
        """
        plot checkerboare squares
        """
        plt.figure()
        plt.imshow(self.squares)

    def generate_sweeps(self):
        """
        generate full screen sweep sequence
        """
        sweepWidth = self.sweepWidth
        stepWidth =  self.stepWidth
        direction = self.direction
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        if direction == "B2U":
            stepY = np.arange(minY - sweepWidth, maxY + stepWidth, stepWidth)
        elif direction == "U2B":
            stepY = np.arange(minY - sweepWidth, maxY + stepWidth, stepWidth)[::-1]
            # stepY = np.arange(maxY, minY - sweepWidth - stepWidth, -1 * stepWidth)
        elif direction == "L2R":
            stepX = np.arange(minX - sweepWidth, maxX + stepWidth, stepWidth)
        elif direction == "R2L":
            stepX = np.arange(minX - sweepWidth, maxX + stepWidth, stepWidth)[::-1]
            # stepX = np.arange(maxX, minX - sweepWidth - stepWidth, -1 * stepWidth)
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
        
        sweepTable = []
        
        if 'stepX' in locals():
            sweeps = np.zeros((len(stepX), np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
            for i in range(len(stepX)):
                temp=sweeps[i,:,:]
                temp[np.where(np.logical_and(mapX >= stepX[i], mapX < (stepX[i] + sweepWidth)))] = 1.0
                sweepTable.append(('V', stepX[i], stepX[i] + sweepWidth))
                del temp
                
        if 'stepY' in locals():
            sweeps = np.zeros((len(stepY), np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
            for j in range(len(stepY)):
                temp=sweeps[j,:,:]
                temp[np.where(np.logical_and(mapY >= stepY[j], mapY < (stepY[j] + sweepWidth)))] = 1.0
                sweepTable.append(('H', stepY[j], stepY[j] + sweepWidth))
                del temp
                
        return sweeps.astype(np.bool), sweepTable

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: square polarity, 1: not reversed; -1: reversed
        third element: sweeps, index in sweep table
        forth element: color of indicator
                       synchronized: gap:-1, sweep on: 1
                       non-synchronized: alternating between -1 and 1 at defined frequency
        for gap frames the second and third elements should be 'None'
        """
        
        sweeps, _ = self.generate_sweeps()
        sweepFrame = self.sweepFrame
        flickerFrame = self.flickerFrame
        iteration = self.iteration
        
        sweepNum = np.size(sweeps,0) # Number of sweeps, vertical or horizontal
        displayFrameNum = sweepFrame * sweepNum # total frame number for the visual stimulation of 1 iteration
        
        #frames for one iteration
        iterFrames=[] 
        
        #add frames for gaps
        for i in range(self.preGapFrameNum):
            iterFrames.append([0,None,None,-1])
        
        
        #add frames for display
        isreverse=[]
        
        for i in range(displayFrameNum):
            
            if (np.floor(i // flickerFrame)) % 2 == 0:
                isreverse = -1
            else:
                isreverse = 1
                
            sweepIndex=int(np.floor(i // sweepFrame))
            
            #add sychornized indicator
            if self.indicator.isSync == True:
                indicatorColor = 1
            else:
                indicatorColor = -1
                
            iterFrames.append([1,isreverse,sweepIndex,indicatorColor])
            
            
        # add gap frames at the end
        for i in range(self.postGapFrameNum):
            iterFrames.append([0,None,None,-1])
        
        fullFrames = []
        
        #add frames for multiple iteration
        for i in range(int(iteration)):
            fullFrames += iterFrames
        
        #add non-synchronized indicator
        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            
            for j in range(np.size(fullFrames,0)):
                if np.floor(j // indicatorFrame) % 2 == 0:
                    fullFrames[j][3] = 1
                else:
                    fullFrames[j][3] = -1
            
        fullFrames = [tuple(x) for x in fullFrames]
        
        return tuple(fullFrames)

    def generate_movie(self):
        """
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        """
        
        self.squares = self.generate_squares()
        
        sweeps, self.sweepTable = self.generate_sweeps()

        self.frames=self.generate_frames()
        
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float16)
        
        for i in range(len(self.frames)):
            currFrame = self.frames[i]
            
            if currFrame[0] == 0:
                currNMsequence = background
                
            else:
                currSquare = self.squares * currFrame[1]
                currSweep = sweeps[currFrame[2]]
                currNMsequence = (currSweep * currSquare) + ((-1 * (currSweep - 1)) * background)

            currNMsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]
            
            fullSequence[i] = currNMsequence
            
            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
        
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')        
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fulldictionary={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicatordict} 
                        
        return fullSequence, fulldictionary

    def clear(self):
        self.sweepTable = None
        self.frames = None
        self.square = None

    def set_direction(self,direction):
        
        if direction == "B2U" or direction == "U2B" or direction == "L2R" or direction == "R2L":
            self.direction = direction
            self.clear()
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'

    def set_sweep_sigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()

    def set_sweep_width(self,sweepWidth):
        self.sweepWidth = sweepWidth
        self.clear()


class NoiseKSstim(Stim):
    """
    obsolete

    generate Kalatsky & Stryker stimulation but with noise movie not flashing 
    squares 
    
    it also integrats flashing indicator for photodiode
    """
    def __init__(self,
                 monitor,
                 indicator,
                 background=0.,
                 coordinate='degree', #'degree' or 'linear'
                 tempFreqCeil = 15, # cutoff temporal frequency (Hz)
                 spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 sweepWidth = 10., # width of sweeps (unit same as Map, cm or deg)
                 sweepSigma=5., # sigma of sweep edges (unit same as Map, cm or deg)
                 sweepEdgeWidth=3., # number of sigmas to smooth the edge of sweeps on each side
                 stepWidth=0.12, # width of steps (unit same as Map, cm or deg)
                 isWarp = False, # warp noise or not
                 direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                 sweepFrame=1, # display frame numbers for each step
                 iteration=1, 
                 preGapDur=2., # gap frame number before flash
                 postGapDur=3., # gap frame number after flash
                 enhanceExp = None): # (0, inf], if smaller than 1, enhance contrast, if bigger than 1, reduce contrast
                     
        super(NoiseKSstim,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate=coordinate,preGapDur=preGapDur,postGapDur=postGapDur)


        self.stimName = 'NoiseKSstim'
        self.tempFreqCeil = tempFreqCeil
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.background = 0
        self.sweepSigma = sweepSigma
        self.sweepWidth = sweepWidth
        self.sweepEdgeWidth = sweepEdgeWidth
        self.isWarp = isWarp
        self.stepWidth = stepWidth
        self.direction = direction
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.enhanceExp = enhanceExp
        
        self.sweepSpeed = self.monitor.refreshRate * self.stepWidth / self.sweepFrame #the speed of sweeps deg/sec

        self.sweepTable = None

        

    def generate_noise_movie(self, frameNum):
        """
        generate filtered noise movie with defined number of frames
        """
        
        Fs_T = self.monitor.refreshRate
        Flow_T = 0
        Fhigh_T = self.tempFreqCeil
        filter_T = generate_filter(frameNum, Fs_T, Flow_T, Fhigh_T, mode = self.filterMode)
        
        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        #print 'Fs_H:', Fs_H
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generate_filter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)
        
        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        #print 'Fs_W:', Fs_W
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generate_filter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)
        
        movie = noise_movie(filter_T, filter_W, filter_H, isplot = False)

        if self.enhanceExp:
                movie = (np.abs(movie)**self.enhanceExp)*(np.copysign(1,movie))
        
        return movie

    def generate_sweeps(self):
        """
        generate full screen sweep sequence
        """
        sweepSigma = self.sweepSigma
        stepWidth =  self.stepWidth
        direction = self.direction
        sweepWidth = float(self.sweepWidth)
        edgeWidth = self.sweepEdgeWidth * self.sweepSigma
        
        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY
            
        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
            
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'
        
        minX = mapX.min()
        maxX = mapX.max()
        
        minY = mapY.min()
        maxY = mapY.max()
        
        if direction == "B2U":
            stepY = np.arange(minY - edgeWidth - sweepWidth / 2, maxY + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)
        elif direction == "U2B":
            stepY = np.arange(minY - edgeWidth - sweepWidth / 2, maxY + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)[::-1]
            # stepY = np.arange(maxY + edgeWidth + sweepWidth / 2, minY - edgeWidth - stepWidth - sweepWidth / 2, -1 * stepWidth)
        elif direction == "L2R":
            stepX = np.arange(minX - edgeWidth - sweepWidth / 2, maxX + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)
        elif direction == "R2L":
            stepX = np.arange(minX - edgeWidth - sweepWidth / 2, maxX + edgeWidth + stepWidth + sweepWidth / 2, stepWidth)[::-1]
            # stepX = np.arange(maxX + edgeWidth + sweepWidth / 2, minX - edgeWidth - stepWidth - sweepWidth / 2, -1 * stepWidth)
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
        
        sweepTable = []
        
        if 'stepX' in locals():
            sweeps = np.ones((len(stepX), np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
            for i in range(len(stepX)):
                currSweep = sweeps[i,:,:]
                
                sweep1 = gaussian(mapX, mu = stepX[i] - sweepWidth / 2, sig = sweepSigma)
                sweep2 = gaussian(mapX, mu = stepX[i] + sweepWidth / 2, sig = sweepSigma)
                
                currSweep[mapX < (stepX[i] - sweepWidth / 2)] = sweep1[mapX < (stepX[i] - sweepWidth / 2)]
                currSweep[mapX > (stepX[i] + sweepWidth / 2)] = sweep2[mapX > (stepX[i] + sweepWidth / 2)]
                
                sweeps[i,:,:] = currSweep
                
                sweepTable.append(('V', stepX[i] - sweepWidth / 2, stepX[i] + sweepWidth / 2))
                
        if 'stepY' in locals():
            sweeps = np.ones((len(stepY), np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
            for j in range(len(stepY)):
                currSweep = sweeps[j,:,:]
                
                sweep1 = gaussian(mapY, mu = stepY[j] - sweepWidth / 2, sig = sweepSigma)
                sweep2 = gaussian(mapY, mu = stepY[j] + sweepWidth / 2, sig = sweepSigma)
                
                currSweep[mapY < (stepY[j] - sweepWidth / 2)] = sweep1[mapY < (stepY[j] - sweepWidth / 2)]
                currSweep[mapY > (stepY[j] + sweepWidth / 2)] = sweep2[mapY > (stepY[j] + sweepWidth / 2)]
                
                sweeps[j,:,:] = currSweep
                
                sweepTable.append(('H', stepY[j] - sweepWidth / 2, stepY[j] + sweepWidth / 2))
                
        return sweeps, sweepTable

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation
        
        returning a list of information of all frames, list of tuples
        
        for each frame:
        
        first element: gap:0 or display:1
        second element: square polarity, None for new KSstim
        third element: sweeps, index in sweep table
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        for gap frames the second and third elements should be 'None'
        """
        
        if not(self.sweepTable):
            _, self.sweepTable = self.generate_sweeps()
        
        sweepTable = self.sweepTable
        sweepFrame = self.sweepFrame
        iteration = self.iteration
        
        sweepNum = len(sweepTable) # Number of sweeps, vertical or horizontal
        displayFrameNum = sweepFrame * sweepNum # total frame number for the visual stimulation of 1 iteration
        
        #frames for one iteration
        iterFrames=[] 
        
        #add frames for gaps
        for i in range(self.preGapFrameNum):
            iterFrames.append([0,None,None,-1])
        
        
        #add frames for display
        
        for i in range(displayFrameNum):
                
            sweepIndex=int(np.floor(i // sweepFrame))
            
            #add sychornized indicator
            if self.indicator.isSync == True:
                indicatorColor = 1
            else:
                indicatorColor = 0
                
            iterFrames.append([1,None,sweepIndex,indicatorColor])
            
            
        # add gap frames at the end
        for i in range(self.postGapFrameNum):
            iterFrames.append([0,None,None,-1])
        
        fullFrames = []
        
        #add frames for multiple iteration
        for i in range(int(iteration)):
            fullFrames += iterFrames
            
        
        #add non-synchronized indicator
        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            
            for j in range(np.size(fullFrames,0)):
                if np.floor(j // indicatorFrame) % 2 == 0:
                    fullFrames[j][3] = 1
                else:
                    fullFrames[j][3] = -1
            
        fullFrames = [tuple(x) for x in fullFrames]
        
        
        return tuple(fullFrames)

    def generate_movie(self):
        """
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        """
        
        sweeps, self.sweepTable = self.generate_sweeps()
        
        self.frames = self.generate_frames()
        
        noise_movie = self.generate_noise_movie(len(self.frames))
        
        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_Lookup_table()
         
        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)
        
        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)
        
        background = np.ones(self.monitor.degCorX.shape, dtype = np.float16) * self.background
        
        for i in range(len(self.frames)):
            currFrame = self.frames[i]
            
            if currFrame[0] == 0:
                currNMsequence = background
            else:
                currImage = noise_movie[i,:,:]
                if self.isWarp:
                    currImage = lookup_image(currImage, lookupI, lookupJ)
                currNMsequence = currImage * sweeps[currFrame[2]]
                
            currNMsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = currNMsequence
            
            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']
        
        
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fullDictionary={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicatordict}
                        
        return fullSequence, fullDictionary

    def clear(self):
        self.sweepTable = None
        self.frames = None
    
    def set_direction(self,direction):
        
        if direction == "B2U" or direction == "U2B" or direction == "L2R" or direction == "R2L":
            self.direction = direction
            self.clear()
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'
            
    def set_sweep_sigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()
        
    def set_sweep_width(self,sweepWidth):
        self.sweepWidth = sweepWidth
        self.clear()


class ObliqueKSstim(Stim):
    """
    obsolete

    generate Kalatsky & Stryker stimulation integrats flashing indicator for
    photodiode
    """
    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree', #'degree' or 'linear'
                 background=0., #back ground color [-1,1]
                 squareSize=25., #size of flickering square
                 squareCenter=(0,0), #coordinate of center point
                 flickerFrame=10,
                 sweepWidth=20., # width of sweeps (unit same as Map, cm or deg)
                 stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                 direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                 sweepFrame=1,
                 iteration=1,
                 preGapDur=2.,
                 postGapDur=3.,
                 rotation_angle=np.pi/4): # the angle of axis rotation

        super(ObliqueKSstim,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate=coordinate,preGapDur=preGapDur,postGapDur=postGapDur)

        self.stimName = 'ObliqueKSstim'
        self.squareSize = squareSize
        self.squareCenter = squareCenter
        self.flickerFrame = flickerFrame
        self.flickerFrequency=self.monitor.refreshRate / self.flickerFrame
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.direction = direction
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.rotation_angle = rotation_angle

        self.sweepSpeed = self.monitor.refreshRate * self.stepWidth / self.sweepFrame #the speed of sweeps deg/sec
        self.flickerHZ = self.monitor.refreshRate / self.flickerFrame

        self.clear()


    def generate_squares(self):
        """
        generate checker board squares
        """


        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY

        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY

        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'

        minX = mapX.min()
        maxX = mapX.max()

        minY = mapY.min()
        maxY = mapY.max()

        negX = np.ceil( abs( ( ( minX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posX = np.ceil( abs( ( ( maxX - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1

        negY = np.ceil( abs( ( ( minY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1
        posY = np.ceil( abs( ( ( maxY - self.squareCenter[0] ) / ( 2 * self.squareSize ) ) ) ) + 1

        squareV = np.ones((np.size(mapX, 0), np.size(mapX, 1)), dtype = np.float16)
        squareV = -1 * squareV

        stepV = np.arange(self.squareCenter[0] - ( 2 * negX + 0.5 ) * self.squareSize,
                          self.squareCenter[0] + ( 2 * posX - 0.5 ) * self.squareSize, self.squareSize*2)

        for i in range(len(stepV)):
            squareV[ np.where( np.logical_and( mapX >= stepV[i], mapX < (stepV[i] + self.squareSize)))] = 1.0

        squareH = np.ones((np.size(mapY, 0), np.size(mapY, 1)), dtype = np.float16)
        squareH = -1 * squareH

        stepH = np.arange(self.squareCenter[1] - ( 2 * negY + 0.5 ) * self.squareSize,
                          self.squareCenter[1] + ( 2 * posY - 0.5 ) * self.squareSize, self.squareSize*2)

        for j in range(len(stepH)):
            squareH[ np.where( np.logical_and( mapY >= stepH[j], mapY < (stepH[j] + self.squareSize)))] = 1

        squares = np.multiply(squareV, squareH)

        return squares

    def plot_squares(self):
        """
        plot checkerboare squares
        """
        plt.figure()
        plt.imshow(self.squares)

    def generate_sweeps(self):
        """
        generate full screen sweep sequence
        """
        sweepWidth = self.sweepWidth
        stepWidth =  self.stepWidth
        direction = self.direction

        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY

        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY

        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'

        all_x = mapX.flatten(); all_y = mapY.flatten()
        rotation_matrix = np.array([[np.cos(self.rotation_angle), np.sin(self.rotation_angle)],
                                    [-np.sin(self.rotation_angle), np.cos(self.rotation_angle)]])

        map_rotated = np.dot(rotation_matrix,np.array([all_x,all_y]))
        map_x_r = map_rotated[0].reshape(mapX.shape); map_y_r = map_rotated[1].reshape(mapY.shape)


        min_x_r = map_x_r.min(); max_x_r = map_x_r.max()
        min_y_r = map_y_r.min(); max_y_r = map_y_r.max()

        if direction == "B2U":
            stepY = np.arange(min_y_r - sweepWidth, max_y_r + stepWidth, stepWidth)
        elif direction == "U2B":
            stepY = np.arange(min_y_r - sweepWidth, max_y_r + stepWidth, stepWidth)[::-1]
            # stepY = np.arange(maxY, minY - sweepWidth - stepWidth, -1 * stepWidth)
        elif direction == "L2R":
            stepX = np.arange(min_x_r - sweepWidth, max_x_r + stepWidth, stepWidth)
        elif direction == "R2L":
            stepX = np.arange(min_x_r - sweepWidth, max_x_r + stepWidth, stepWidth)[::-1]
            # stepX = np.arange(maxX, minX - sweepWidth - stepWidth, -1 * stepWidth)
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'

        sweepTable = []

        if 'stepX' in locals():
            sweeps = np.zeros((len(stepX), np.size(map_x_r, 0), np.size(map_x_r, 1)), dtype = np.float16)
            for i in range(len(stepX)):
                temp=sweeps[i,:,:]
                temp[np.where(np.logical_and(map_x_r >= stepX[i], map_x_r < (stepX[i] + sweepWidth)))] = 1.0
                sweepTable.append(('V', stepX[i], stepX[i] + sweepWidth))
                del temp

        if 'stepY' in locals():
            sweeps = np.zeros((len(stepY), np.size(map_y_r, 0), np.size(map_y_r, 1)), dtype = np.float16)
            for j in range(len(stepY)):
                temp=sweeps[j,:,:]
                temp[np.where(np.logical_and(map_y_r >= stepY[j], map_y_r < (stepY[j] + sweepWidth)))] = 1.0
                sweepTable.append(('H', stepY[j], stepY[j] + sweepWidth))
                del temp

        return sweeps.astype(np.bool), sweepTable

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: square polarity, 1: not reversed; -1: reversed
        third element: sweeps, index in sweep table
        forth element: color of indicator
                       synchronized: gap:0, then alternating between -1 and 1 for each sweep
                       non-synchronized: alternating between -1 and 1 at defined frequency
        for gap frames the second and third elements should be 'None'
        """

        sweeps, _ = self.generate_sweeps()
        sweepFrame = self.sweepFrame
        flickerFrame = self.flickerFrame
        iteration = self.iteration

        sweepNum = np.size(sweeps,0) # Number of sweeps, vertical or horizontal
        displayFrameNum = sweepFrame * sweepNum # total frame number for the visual stimulation of 1 iteration

        #frames for one iteration
        iterFrames=[]

        #add frames for gaps
        for i in range(self.preGapFrameNum):
            iterFrames.append([0,None,None,-1])


        #add frames for display
        isreverse=[]

        for i in range(displayFrameNum):

            if (np.floor(i // flickerFrame)) % 2 == 0:
                isreverse = -1
            else:
                isreverse = 1

            sweepIndex=int(np.floor(i // sweepFrame))

            #add sychornized indicator
            if self.indicator.isSync == True:
                indicatorColor = 1
            else:
                indicatorColor = -1

            iterFrames.append([1,isreverse,sweepIndex,indicatorColor])


        # add gap frames at the end
        for i in range(self.postGapFrameNum):
            iterFrames.append([0,None,None,-1])

        fullFrames = []

        #add frames for multiple iteration
        for i in range(int(iteration)):
            fullFrames += iterFrames

        #add non-synchronized indicator
        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum

            for j in range(np.size(fullFrames,0)):
                if np.floor(j // indicatorFrame) % 2 == 0:
                    fullFrames[j][3] = 1
                else:
                    fullFrames[j][3] = -1

        fullFrames = [tuple(x) for x in fullFrames]

        return tuple(fullFrames)

    def generate_movie(self):
        """
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        """

        self.squares = self.generate_squares()

        sweeps, self.sweepTable = self.generate_sweeps()

        self.frames=self.generate_frames()

        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float16)

        for i in range(len(self.frames)):
            currFrame = self.frames[i]

            if currFrame[0] == 0:
                currNMsequence = background

            else:
                currSquare = self.squares * currFrame[1]
                currSweep = sweeps[currFrame[2]]
                currNMsequence = (currSweep * currSquare) + ((-1 * (currSweep - 1)) * background)

            currNMsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = currNMsequence

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']


        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fulldictionary={'stimulation':KSdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fulldictionary

    def clear(self):
        self.sweepTable = None
        self.frames = None
        self.square = None

    def set_direction(self,direction):

        if direction == "B2U" or direction == "U2B" or direction == "L2R" or direction == "R2L":
            self.direction = direction
            self.clear()
        else:
            raise LookupError, 'attribute "direction" should be "B2U", "U2B", "L2R" or "R2L".'

    def set_sweep_sigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()

    def set_sweep_width(self,sweepWidth):
        self.sweepWidth = sweepWidth
        self.clear()


class FlashingNoise(Stim):

    """
    obsolete

    generate flashing full field noise with background displayed before and after

    it also integrats flashing indicator for photodiode
    """

    def __init__(self,
                 monitor,
                 indicator,
                 coordinate = 'degree', #'degree' or 'linear'
                 background=0.,
                 spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 iteration=1, # time to flash
                 flashFrameNum=1, # frame number for display noise of each flash
                 preGapDur=2., # gap frame number before flash
                 postGapDur=3., # gap frame number after flash
                 isWarp = False): # warp noise or not

        super(FlashingNoise,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate=coordinate,preGapDur=preGapDur,postGapDur=postGapDur)

        self.stimName = 'FlashingNoise'
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.iteration = iteration
        self.flashFrameNum = flashFrameNum
        self.isWarp = isWarp

    def generate_noise_movie(self):
        """
        generate filtered noise movie with defined number of frames
        """

        frameNum = self.flashFrameNum * self.iteration
        filter_T = np.ones((frameNum))

        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generate_filter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)

        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generate_filter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)

        movie = noise_movie(filter_T, filter_W, filter_H, isplot = False)

        return movie

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        """

        #frame number for each iteration
        iterationFrameNum = self.preGapFrameNum+self.flashFrameNum+self.postGapFrameNum

        frames = np.zeros((self.iteration*(iterationFrameNum),4)).astype(np.int)

        #initilize indicator color
        frames[:,3] = -1

        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i,2] = i // iterationFrameNum

            # mark start frame of every iteration
            if i % iterationFrameNum == 0:
                frames[i, 1] = 1

            # mark display frame and synchronized indicator
            if ((i % iterationFrameNum >= self.preGapFrameNum) and \
               (i % iterationFrameNum < (self.preGapFrameNum + self.flashFrameNum))):

                frames[i, 0] = 1

                if self.indicator.isSync:
                    frames[i, 3] = 1

            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i, 3] = 1
                else:
                    frames[i, 3] = -1

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generating movie
        """

        self.frames = self.generate_frames()
        noise_movie = self.generate_noise_movie()

        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_Lookup_table()

        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float16)

        for i in range(len(self.frames)):
            currFrame = self.frames[i]

            if currFrame[0] == 0:
                currFNsequence = background
            else:
                currFNsequence = noise_movie[currFrame[2],:,:]
                if self.isWarp:
                    currFNsequence = lookup_image(currFNsequence, lookupI, lookupJ)

            currFNsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = currFNsequence

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']

        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        fullDictionary={'stimulation':NFdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fullDictionary

    def set_flash_frame_num(self, flashFrameNum):
        self.flashFrameNum = flashFrameNum
        self.clear()


class GaussianNoise(Stim):
    """
    obsolete

    generate full field noise movie with contrast modulated by gaussian function
    """
    def __init__(self,
                 monitor,
                 indicator,
                 background = 0.,
                 coordinate = 'degree', #'degree' or 'linear'
                 tempFreqCeil = 15, # cutoff temporal frequency (Hz)
                 spatialFreqCeil = 0.02, # cutoff spatial frequency (cycle/degree)
                 filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                 sweepSigma=10., # sigma of sweep edges (unit same as Map, cm or deg)
                 sweepWidth=10., # width of sweeps (unit same as Map, cm or deg)
                 sweepEdgeWidth=3., # number of sigmas to smooth the edge of sweeps on each side
                 stepWidth=0.12, # width of steps (unit same as Map, cm or deg)
                 sweepFrame=1, # display frame numbers for each step
                 iteration=2,
                 preGapDur=2., # gap frame number before flash
                 postGapDur=3., # gap frame number after flash
                 isWarp = False, # warp noise or not
                 contrast = 0.5, # contrast of the movie from 0 to 1
                 enhanceExp = None): # (0, inf], if smaller than 1, enhance contrast, if bigger than 1, reduce contrast

        super(GaussianNoise,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate=coordinate,preGapDur=preGapDur,postGapDur=postGapDur)


        self.stimName = 'GaussianNoise'
        self.tempFreqCeil = tempFreqCeil
        self.spatialFreqCeil = spatialFreqCeil
        self.filterMode = filterMode
        self.sweepSigma = sweepSigma
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.sweepEdgeWidth= sweepEdgeWidth
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.isWarp = isWarp
        self.contrast = contrast
        self.enhanceExp = enhanceExp



    def generate_noise_movie(self, frameNum):
        """
        generate filtered noise movie with defined number of frames
        """

        Fs_T = self.monitor.refreshRate
        Flow_T = 0
        Fhigh_T = self.tempFreqCeil
        filter_T = generate_filter(frameNum, Fs_T, Flow_T, Fhigh_T, mode = self.filterMode)

        hPixNum = self.monitor.resolution[0]/self.monitor.downSampleRate
        pixHeightCM = self.monitor.monHcm / hPixNum
        Fs_H = 1 / (np.arcsin(pixHeightCM / self.monitor.dis) * 180 /  np.pi)
        #print 'Fs_H:', Fs_H
        Flow_H = 0
        Fhigh_H = self.spatialFreqCeil
        filter_H = generate_filter(hPixNum, Fs_H, Flow_H, Fhigh_H, mode = self.filterMode)

        wPixNum = self.monitor.resolution[1]/self.monitor.downSampleRate
        pixWidthCM = self.monitor.monWcm / wPixNum
        Fs_W = 1 / (np.arcsin(pixWidthCM / self.monitor.dis) * 180 / np.pi)
        #print 'Fs_W:', Fs_W
        Flow_W = 0
        Fhigh_W = self.spatialFreqCeil
        filter_W = generate_filter(wPixNum, Fs_W, Flow_W, Fhigh_W, mode = self.filterMode)

        movie = noise_movie(filter_T, filter_W, filter_H, isplot = False)

        if self.enhanceExp:
                movie = (np.abs(movie)**self.enhanceExp)*(np.copysign(1,movie))

        return movie

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration
        forth element: color of indicator, gap:0, then alternating between -1 and 1 for each sweep
        fifth element: if is display, the contrast
        """

        sweepEdge = self.sweepEdgeWidth * self.sweepSigma

        steps = np.arange(-sweepEdge - self.sweepWidth / 2, sweepEdge + self.sweepWidth / 2, self.stepWidth)

        displayFrameNum = self.sweepFrame * len(steps) # total frame number for the visual stimulation of 1 iteration

        stepContrast = np.ones(len(steps))
        stepContrast1 = gaussian(steps, mu=-self.sweepWidth / 2, sig=self.sweepSigma)
        stepContrast2 = gaussian(steps, mu=+self.sweepWidth / 2, sig=self.sweepSigma)

        stepContrast[steps < (-self.sweepWidth / 2)] = stepContrast1[steps < (-self.sweepWidth / 2)]
        stepContrast[steps > (self.sweepWidth / 2)] = stepContrast2[steps > (self.sweepWidth / 2)]


        #frame number for each iteration
        iterationFrameNum = self.preGapFrameNum + displayFrameNum + self.postGapFrameNum

        frames = np.zeros((self.iteration*(iterationFrameNum),5)).astype(np.float32)

        #initilize indicator color
        frames[:,3] = -1

        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i,2] = i // iterationFrameNum

            # mark start frame of every iteration
            if i % iterationFrameNum == 0:
                frames[i, 1] = 1

            # mark display frame and synchronized indicator
            if ((i % iterationFrameNum >= self.preGapFrameNum) and \
               (i % iterationFrameNum < (self.preGapFrameNum + displayFrameNum))):

                frames[i, 0] = 1

                if self.indicator.isSync:
                    frames[i, 3] = 1

            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i,3] = 1
                else:
                    frames[i,3] = -1

            # mark display contrast
            currFrameNumInIteration = i % iterationFrameNum
            if (currFrameNumInIteration < self.preGapFrameNum) or \
               (currFrameNumInIteration >= self.preGapFrameNum + displayFrameNum):
                frames[i,4] = np.nan
            else:
                displayInd = currFrameNumInIteration - self.preGapFrameNum
                frames[i,4] = stepContrast[displayInd // self.sweepFrame] * self.contrast

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generating movie
        """

        self.frames = self.generate_frames()
        iterationFrameNum = len(self.frames) / self.iteration


        if self.isWarp:
            lookupI, lookupJ = self.monitor.generate_Lookup_table()

        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        background = self.background * np.ones((self.monitor.degCorX.shape), dtype = np.float16)

        for i in range(len(self.frames)):

            currFrame = self.frames[i]

            if currFrame[1] == 1:
                displayFrameNum = iterationFrameNum - self.preGapFrameNum - self.postGapFrameNum
                noise_movie = self.generate_noise_movie(displayFrameNum)

            if currFrame[0] == 0:
                currGNsequence = background
            else:
                currDisplayInd = (i % iterationFrameNum) - self.preGapFrameNum
                currGNsequence = noise_movie[currDisplayInd,:,:] * currFrame[4]
                if self.isWarp:
                    currGNsequence = lookup_image(currGNsequence, lookupI, lookupJ)

            currGNsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = currGNsequence

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']

        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        fullDictionary={'stimulation':KSdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fullDictionary

    def set_flash_frame_num(self, flashFrameNum):
        self.flashFrame = flashFrameNum
        self.clear()

    def set_sweepSigma(self, sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()

    def set_sweepWidth(self, sweepWidth):
        self.sweepWidth = sweepWidth
        self.clear()

    def set_contrast(self, contrast):
        self.contrast = contrast
        self.clear()


class FlashingCircle(Stim):
    """
    flashing circle stimulation.
    """

    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree', # 'degree' or 'linear'
                 center = (90., 10.), # center coordinate of the circle (degree)
                 radius = 10., # radius of the circle
                 color = -1., # color of the circle [-1: 1]
                 iteration= 1, # total number of flashes
                 flashFrame= 3, # frame number for display circle of each flash
                 preGapDur=2., # gap frame number before flash
                 postGapDur=3., # gap frame number after flash
                 background = 0.):

        super(FlashingCircle,self).__init__(monitor=monitor, indicator=indicator, background=background,
                                            coordinate=coordinate, preGapDur=preGapDur, postGapDur=postGapDur)

        self.stimName = 'FlashingCircle'
        self.center = center
        self.radius = radius
        self.color = color
        self.iteration = iteration
        self.flashFrame = flashFrame
        self.frameConfig = ('isDisplay', 'isIterationStart', 'currentIteration', 'indicatorColor')

        self.clear()

    def set_flash_frame_num(self, flashFrameNum):
        self.flashFrame = flashFrameNum
        self.clear()

    def set_color(self, color):
        self.color = color
        self.clear()

    def set_center(self, center):
        self.center = center
        self.clear()

    def set_radius(self, radius):
        self.radius = radius
        self.clear()

    def generate_frames(self):
        """
        function to generate all the frames needed for the stimulation

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: iteration start, first frame of each iteration: 1; other frames: 0
        third element: current iteration
        forth element: color of indicator, gap:0, display:1
        """

        #frame number for each iteration
        iterationFrameNum = self.preGapFrameNum+self.flashFrame+self.postGapFrameNum

        frames = np.zeros((self.iteration*(iterationFrameNum),4)).astype(np.int)

        #initilize indicator color
        frames[:,3] = -1

        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i, 2] = i // iterationFrameNum

            # mark start frame of every iteration
            if i % iterationFrameNum == 0:
                frames[i, 1] = 1

            # mark display frame and synchronized indicator
            if ((i % iterationFrameNum >= self.preGapFrameNum) and \
               (i % iterationFrameNum < (self.preGapFrameNum + self.flashFrame))):

                frames[i, 0] = 1

                if self.indicator.isSync:
                    frames[i, 3] = 1

            # mark unsynchronized indicator
            if not(self.indicator.isSync):
                if np.floor(i // self.indicator.frameNum) % 2 == 0:
                    frames[i, 3] = 1
                else:
                    frames[i, 3] = -1

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generating movie
        """

        self.frames = self.generate_frames()

        fullSequence = np.zeros((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16)

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        background = self.background * np.ones((np.size(self.monitor.degCorX, 0), np.size(self.monitor.degCorX,1)), dtype = np.float16)

        if self.coordinate == 'degree':
            mapX = self.monitor.degCorX
            mapY = self.monitor.degCorY

        elif self.coordinate == 'linear':
            mapX = self.monitor.linCorX
            mapY = self.monitor.linCorY
        else:
            raise LookupError, 'the "coordinate" attributate show be either "degree" or "linear"'

        circleMask = circle_mask(mapX,mapY,self.center,self.radius).astype(np.float16)

        for i in range(len(self.frames)):
            currFrame = self.frames[i]

            if currFrame[0] == 0:
                currFCsequence = background
            else:
                currFCsequence = (circleMask * self.color) + ((-1 * (circleMask - 1)) * background)

            currFCsequence[indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[3]

            fullSequence[i] = currFCsequence

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']

        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        fullDictionary={'stimulation':NFdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fullDictionary


class SparseNoise(Stim):
    """
    generate sparse noise stimulus integrates flashing indicator for photodiode
    """

    def __init__(self,
                 monitor,
                 indicator,
                 background=0., #back ground color [-1,1]
                 coordinate='degree', #'degree' or 'linear'
                 gridSpace=(10.,10.), #(alt,azi)
                 probeSize=(10.,10.), #size of flicker probes (width,height)
                 probeOrientation=0., #orientation of flicker probes
                 probeFrameNum=3, #number of frames for each square presentation
                 subregion=None, #[minAlt, maxAlt, minAzi, maxAzi]
                 sign='ON-OFF', # 'On', 'OFF' or 'ON-OFF'
                 iteration=1,
                 preGapDur=2.,
                 postGapDur=3.):

        super(SparseNoise,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate = coordinate,preGapDur=preGapDur,postGapDur=postGapDur)


        self.stimName = 'SparseNoise'
        self.gridSpace = gridSpace
        self.probeSize = probeSize
        self.probeOrientation = probeOrientation
        self.probeFrameNum = probeFrameNum
        self.frameConfig = ('isDisplay', '(azimuth, altitude)', 'polarity', 'indicatorColor')

        if subregion is None:
            if self.coordinate == 'degree':
                self.subregion = [np.amin(self.monitor.degCorY), np.amax(self.monitor.degCorY),
                                  np.amin(self.monitor.degCorX), np.amax(self.monitor.degCorX)]
            if self.coordinate == 'linear':
                self.subregion = [np.amin(self.monitor.linCorY), np.amax(self.monitor.linCorY),
                                  np.amin(self.monitor.linCorX), np.amax(self.monitor.linCorX)]
        else:
            self.subregion = subregion

        self.sign = sign
        self.iteration = iteration

        self.clear()

    def _getGridPoints(self):
        """
        generate all the grid points in display area (subregion and monitor coverage)
        [azi, alt]
        """

        rows = np.arange(self.subregion[0], self.subregion[1] + self.gridSpace[0], self.gridSpace[0])
        columns = np.arange(self.subregion[2], self.subregion[3] + self.gridSpace[1], self.gridSpace[1])

        xx,yy = np.meshgrid(columns,rows)

        gridPoints = np.transpose(np.array([xx.flatten(),yy.flatten()]))

        #get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitorPoints = np.transpose(np.array([self.monitor.degCorX.flatten(),self.monitor.degCorY.flatten()]))
        if self.coordinate == 'linear':
            monitorPoints = np.transpose(np.array([self.monitor.linCorX.flatten(),self.monitor.linCorY.flatten()]))

        #get the grid points within the coverage of monitor
        gridPoints = gridPoints[in_hull(gridPoints,monitorPoints)]

        return gridPoints

    def _generate_grid_points_sequence(self):
        """
        generate pseudorandomized grid point sequence. if ON-OFF, consecutive frames should not
        present stimulus at same location
        :return: list of [gridPoint, sign]
        """

        gridPoints = self._getGridPoints()

        if self.sign == 'ON':
            gridPoints = [[x,1] for x in gridPoints]
            shuffle(gridPoints)
            return gridPoints
        elif self.sign == 'OFF':
            gridPoints = [[x,-1] for x in gridPoints]
            shuffle(gridPoints)
            return gridPoints
        elif self.sign == 'ON-OFF':
            allGridPoints = [[x,1] for x in gridPoints] + [[x,-1] for x in gridPoints]
            shuffle(allGridPoints)
            # remove coincident hit of same location by continuous frames
            print 'removing coincident hit of same location with continuous frames:'
            while True:
                iteration = 0
                coincidentHitNum = 0
                for i, gridPoint in enumerate(allGridPoints[:-3]):
                    if (allGridPoints[i][0] == allGridPoints[i+1][0]).all():
                        allGridPoints[i+1], allGridPoints[i+2] = allGridPoints[i+2], allGridPoints[i+1]
                        coincidentHitNum += 1
                iteration += 1
                print 'iteration:',iteration,'  continous hits number:',coincidentHitNum
                if coincidentHitNum == 0:
                    break

            return allGridPoints

    def generate_frames(self):
        """
        function to generate all the frames needed for SparseNoiseStimu

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: tuple, retinotopic location of the center of current square,[azi,alt]
        third element: polarity of current square, 1: bright, -1: dark
        forth element: color of indicator
                       synchronized: gap:0, 1 for onset frame for each square, -1 for the rest
                       non-synchronized: alternating between -1 and 1 at defined frequency
        for gap frames the second and third elements should be 'None'
        """

        frames = []
        if self.probeFrameNum == 1:
            indicatorONFrame = 1
        elif self.probeFrameNum >1:
            indicatorONFrame = self.probeFrameNum // 2
        else:
            raise ValueError('self.probeFrameNum should be an integer larger than 0!')

        indicatorOFFFrame = self.probeFrameNum - indicatorONFrame

        for i in range(self.iteration):

            if self.preGapFrameNum>0: frames += [[0,None,None,-1]]*self.preGapFrameNum

            iterGridPoints = self._generate_grid_points_sequence()

            for gridPoint in iterGridPoints:
                frames += [[1,gridPoint[0],gridPoint[1],1]] * indicatorONFrame
                frames += [[1,gridPoint[0],gridPoint[1],-1]] * indicatorOFFFrame

            if self.postGapFrameNum>0: frames += [[0,None,None,-1]]*self.postGapFrameNum

        if self.indicator.isSync == False:
            indicatorFrame = self.indicator.frameNum
            for m in range(len(frames)):
                if np.floor(m // indicatorFrame) % 2 == 0:
                    frames[m][3] = 1
                else:
                    frames[m][3] = -1

        frames=tuple(frames)

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generate movie for display
        """

        self.frames = self.generate_frames()

        if self.coordinate=='degree':corX=self.monitor.degCorX;corY=self.monitor.degCorY
        elif self.coordinate=='linear':corX=self.monitor.linCorX;corY=self.monitor.linCorY

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        fullSequence = np.ones((len(self.frames),self.monitor.degCorX.shape[0],self.monitor.degCorX.shape[1]),dtype=np.float16) * self.background

        for i, currFrame in enumerate(self.frames):
            if currFrame[0] == 1: # not a gap
                if i == 0: # first frame and (not a gap)
                    currDisplayMatrix = get_warped_square(corX, corY, center = currFrame[1], width=self.probeSize[0],
                                                          height=self.probeSize[1], ori=self.probeOrientation,
                                                          foregroundColor=currFrame[2], backgroundColor=self.background)
                else: # (not first frame) and (not a gap)
                    if self.frames[i-1][1] is None: # (not first frame) and (not a gap) and (new square from gap)
                        currDisplayMatrix = get_warped_square(corX, corY, center = currFrame[1], width=self.probeSize[0],
                                                              height=self.probeSize[1], ori=self.probeOrientation,
                                                              foregroundColor=currFrame[2], backgroundColor=self.background)
                    elif (currFrame[1]!=self.frames[i-1][1]).any() or (currFrame[2]!=self.frames[i-1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        currDisplayMatrix = get_warped_square(corX, corY, center = currFrame[1], width=self.probeSize[0],
                                                              height=self.probeSize[1], ori=self.probeOrientation,
                                                              foregroundColor=currFrame[2], backgroundColor=self.background)

                #assign current display matrix to full sequence
                fullSequence[i] = currDisplayMatrix

            #add sync square for photodiode
            fullSequence[i, indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax]=currFrame[3]

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']

        #generate log dictionary
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        SNdict=dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        fulldictionary={'stimulation':SNdict,
                        'monitor':mondict,
                        'indicator':indicatordict}

        return fullSequence, fulldictionary


class DriftingGratingCircle(Stim):
    """
    class of drifting grating circle stimulus
    """

    def __init__(self,
                 monitor,
                 indicator,
                 background=0., # back ground color [-1,1]
                 coordinate='degree', # 'degree' or 'linear'
                 center=(60.,0.), # (azi, alt), unit defined by self.coordinate
                 sf_list=(0.08,), # (0.16,0.08,0.04), spatial frequency, cycle/unit
                 tf_list=(4.,), # (15.,4.,0.5), temporal frequency, Hz
                 dire_list=(0.,), # np.arange(0,2*np.pi,np.pi/2), direction, arc
                 con_list=(0.5,), # (0.01,0.02,0.05,0.11,0.23,0.43,0.73,0.95), contrast, [0, 1]
                 size_list=(5.,), # (1.,2.,5.,10.), radius of the circle, unit defined by self.coordinate
                 blockDur=2., # duration of each condition, second
                 midGapDur=0.5, # duration of gap between conditions
                 iteration=2, # iteration of whole sequence
                 preGapDur=2.,
                 postGapDur=3.):

        super(DriftingGratingCircle,self).__init__(monitor=monitor,indicator=indicator,background=background,coordinate=coordinate,preGapDur=preGapDur,postGapDur=postGapDur)

        self.stimName = 'DriftingGratingCircle'
        self.center = center
        self.sf_list = sf_list
        self.tf_list = tf_list
        self.dire_list = dire_list
        self.con_list = con_list
        self.size_list = size_list
        self.blockDur = blockDur
        self.midGapDur = midGapDur
        self.iteration = iteration
        self.frameConfig = ('isDisplay', 'isCycleStart', 'spatialFrequency', 'temporalFrequency', 'direction',
                            'contrast', 'radius', 'phase', 'indicatorColor')

        for tf in tf_list:
            period = 1. / tf
            if (0.05 * period) < (blockDur % period) < (0.95 * period):
                print period
                print blockDur % period
                print 0.95 * period
                error_msg = 'Duration of each block times tf '+ str(tf) + ' should be close to a whole number!'
                raise ValueError, error_msg

    def _generate_all_conditions(self):
        """
        generate all possible conditions for one iteration given the lists of parameters
        :return:
        """
        all_conditions = [(sf, tf, dire, con, size) for sf in self.sf_list
                                                    for tf in self.tf_list
                                                    for dire in self.dire_list
                                                    for con in self.con_list
                                                    for size in self.size_list]
        random.shuffle(all_conditions)
        # print ['sf', 'tf', 'dire', 'con', 'size']
        # print '\n'.join([str(condi) for condi in all_conditions])

        return all_conditions

    def _generate_phase_list(self, tf):
        """

        get a list of phases will be displayed for each frame in the block duration, also make the first frame of each
        cycle

        :param tf: temporal frequency
        :return: list of phases in one block, number of frames for each circle
        """

        block_frame_num = int(self.blockDur * self.monitor.refreshRate)

        frame_per_cycle = int(self.monitor.refreshRate / tf)

        phaces_per_cycle = list(np.arange(0,np.pi*2,np.pi*2/frame_per_cycle))

        phases = []

        while len(phases) < block_frame_num:
            phases += phaces_per_cycle

        phases = phases[0:block_frame_num]

        return phases, frame_per_cycle

    @staticmethod
    def _get_ori(dire):
        """
        get orientation from direction, [0, pi)
        """
        return (dire + np.pi / 2) % np.pi

    def generate_frames(self):
        """
        function to generate all the frames needed for DriftingGratingCircle

        returning a list of information of all frames, list of tuples

        for each frame:

        first element: gap:0 or display:1
        second element: first frame in a cycle:1; rest:0
        third element: spatial frequency
        forth element: temporal frequency
        fifth element: direction, [0, 2*pi)
        sixth element: contrast
        seventh element: size (raidus of the circle)
        eighth element: phase, [0, 2*pi)
        ninth element: indicator color [-1, 1], gap:-1, first frame of cycle:1, rest frames of cycle: 0
        for gap frames from the second to the eighth elements should be 'None'
        """

        frames = []

        for i in range(self.iteration):
            if i == 0: # very first block
                frames += [[0, None,None,None,None,None,None,None,-1.] for ind in range(self.preGapFrameNum)]
            else: # first block for the later iteration
                frames += [[0, None,None,None,None,None,None,None,-1.] for ind in range(int(self.midGapDur * self.monitor.refreshRate))]

            all_conditions = self._generate_all_conditions()

            for j, condition in enumerate(all_conditions):
                if j != 0: # later conditions
                    frames += [[0, None,None,None,None,None,None,None,-1.] for ind in range(int(self.midGapDur * self.monitor.refreshRate))]

                sf, tf, dire, con, size = condition

                # get phase list for each condition
                phases, frame_per_cycle = self._generate_phase_list(tf)
                if (dire % (np.pi * 2)) >= np.pi: phases = [-phase for phase in phases]

                for k, phase in enumerate(phases): # each frame in the block

                    # mark first frame of each cycle
                    if k % frame_per_cycle == 0:
                        first_in_cycle = 1
                    else:
                        first_in_cycle = 0

                    frames.append([1,first_in_cycle,sf,tf,dire,con,size,phase,float(first_in_cycle)])

        # add post gap frame
        frames += [[0, None,None,None,None,None,None,None,-1.] for ind in range(self.postGapFrameNum)]

        #add non-synchronized indicator
        if self.indicator.isSync == False:
            for l in range(len(frames)):
                if np.floor(l // self.indicator.frameNum) % 2 == 0:
                    frames[l][-1] = 1
                else:
                    frames[l][-1] = -1

        # switch each frame to tuple
        frames = [tuple(frame) for frame in frames]

        return tuple(frames)
    
    def _generate_circle_mask_dict(self):
        """
        generate a dictionary of circle masks for each size in size list
        """
        
        masks = {}
        if self.coordinate=='degree':corX=self.monitor.degCorX;corY=self.monitor.degCorY
        elif self.coordinate=='linear':corX=self.monitor.linCorX;corY=self.monitor.linCorY
        
        for size in self.size_list:
            curr_mask = circle_mask(corX, corY, self.center, size)
            masks.update({size:curr_mask})
            
        return masks

    def generate_movie(self):
        
        
        self.frames = self.generate_frames()
        mask_dict = self._generate_circle_mask_dict()

        if self.coordinate=='degree':corX=self.monitor.degCorX;corY=self.monitor.degCorY
        elif self.coordinate=='linear':corX=self.monitor.linCorX;corY=self.monitor.linCorY
        else:
            raise LookupError, "self.coordinate should be either 'linear' or 'degree'."

        indicatorWmin=self.indicator.centerWpixel - (self.indicator.width_pixel / 2)
        indicatorWmax=self.indicator.centerWpixel + (self.indicator.width_pixel / 2)
        indicatorHmin=self.indicator.centerHpixel - (self.indicator.height_pixel / 2)
        indicatorHmax=self.indicator.centerHpixel + (self.indicator.height_pixel / 2)

        mov = np.ones((len(self.frames),corX.shape[0],corX.shape[1]),dtype=np.float16) * self.background
        background_frame = np.ones(corX.shape,dtype=np.float16) * self.background

        for i, currFrame in enumerate(self.frames):

            if currFrame[0] == 1: # not a gap

                curr_ori = self._get_ori(currFrame[4])

                curr_grating = get_grating(corX,
                                           corY,
                                           ori = curr_ori,
                                           spatial_freq = currFrame[2],
                                           center = self.center,
                                           phase = currFrame[7],
                                           contrast = currFrame[5])
                curr_grating = curr_grating * 2. - 1.

                curr_circle_mask = mask_dict[currFrame[6]]

                mov[i] = (curr_grating * curr_circle_mask) + (background_frame * (curr_circle_mask * -1. + 1.))

            #add sync square for photodiode
            mov[i, indicatorHmin:indicatorHmax, indicatorWmin:indicatorWmax] = currFrame[-1]


            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) / len(self.frames)))+'%']



        #generate log dictionary
        mondict=dict(self.monitor.__dict__)
        indicatordict=dict(self.indicator.__dict__)
        indicatordict.pop('monitor')
        self_dict=dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')
        log={'stimulation':self_dict,
             'monitor':mondict,
             'indicator':indicatordict}

        return mov, log

   
class KSstimAllDir(object):
    """
    generate Kalatsky & Stryker stimulation in all four direction contiuously
    """
    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree', #'degree' or 'linear'
                 background=0., #back ground color [-1,1]
                 squareSize=25, #size of flickering square
                 squareCenter=(0,0), #coordinate of center point
                 flickerFrame=6,
                 sweepWidth=20., # width of sweeps (unit same as Map, cm or deg)
                 stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                 sweepFrame=1,
                 iteration=1,
                 preGapDur=2.,
                 postGapDur=3.):

        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate
        self.squareSize = squareSize
        self.squareCenter = squareCenter
        self.flickerFrame = flickerFrame
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.preGapDur = preGapDur
        self.postGapDur = postGapDur


    def generate_movie(self):

        KS_stim=KSstim(self.monitor,
                       self.indicator,
                       background=self.background,
                       coordinate=self.coordinate,
                       direction='B2U',
                       squareSize=self.squareSize,
                       squareCenter=self.squareCenter,
                       flickerFrame=self.flickerFrame,
                       sweepWidth=self.sweepWidth,
                       stepWidth=self.stepWidth,
                       sweepFrame=self.sweepFrame,
                       iteration=self.iteration,
                       preGapDur=self.preGapDur,
                       postGapDur=self.postGapDur)

        movB2U, dictB2U = KS_stim.generate_movie()
        KS_stim.set_direction('U2B')
        movU2B, dictU2B = KS_stim.generate_movie()
        KS_stim.set_direction('L2R')
        movL2R, dictL2R = KS_stim.generate_movie()
        KS_stim.set_direction('R2L')
        movR2L, dictR2L = KS_stim.generate_movie()

        mov = np.vstack((movB2U,movU2B,movL2R,movR2L))
        log = {'monitor':dictB2U['monitor'],
               'indicator':dictB2U['indicator']}
        stimulation = dict(dictB2U['stimulation'])
        stimulation['stimName'] = 'KSstimAllDir'
        stimulation['direction'] = ['B2U','U2B','L2R','R2L']

        sweepTable = []
        frames = []

        sweepTableB2U = dictB2U['stimulation']['sweepTable']; framesB2U = dictB2U['stimulation']['frames']; sweepLenB2U = len(sweepTableB2U)
        sweepTableB2U = [ ['B2U', x[1], x[2]] for x in sweepTableB2U]; framesB2U = [[x[0],x[1],x[2],x[3],'B2U'] for x in framesB2U]
        sweepTable += sweepTableB2U; frames += framesB2U

        sweepTableU2B = dictU2B['stimulation']['sweepTable']; framesU2B = dictU2B['stimulation']['frames']; sweepLenU2B = len(sweepTableU2B)
        sweepTableU2B = [ ['U2B', x[1], x[2]] for x in sweepTableU2B]; framesU2B = [[x[0],x[1],x[2],x[3],'U2B'] for x in framesU2B]
        for frame in framesU2B:
            if frame[2] is not None: frame[2] += sweepLenB2U
        sweepTable += sweepTableU2B; frames += framesU2B

        sweepTableL2R = dictL2R['stimulation']['sweepTable']; framesL2R = dictL2R['stimulation']['frames']; sweepLenL2R = len(sweepTableL2R)
        sweepTableL2R = [ ['L2R', x[1], x[2]] for x in sweepTableL2R]; framesL2R = [[x[0],x[1],x[2],x[3],'L2R'] for x in framesL2R]
        for frame in framesL2R:
            if frame[2] is not None: frame[2] += sweepLenB2U+sweepLenU2B
        sweepTable += sweepTableL2R; frames += framesL2R

        sweepTableR2L = dictR2L['stimulation']['sweepTable']; framesR2L = dictR2L['stimulation']['frames']
        sweepTableR2L = [ ['R2L', x[1], x[2]] for x in sweepTableR2L]; framesR2L = [[x[0],x[1],x[2],x[3],'R2L'] for x in framesR2L]
        for frame in framesR2L:
            if frame[2] is not None: frame[2] += sweepLenB2U+sweepLenU2B+sweepLenL2R
        sweepTable += sweepTableR2L; frames += framesR2L

        stimulation['frames'] = [tuple(x) for x in frames]
        stimulation['sweepTable'] = [tuple(x) for x in sweepTable]

        log['stimulation'] = stimulation
        log['stimulation']['frameConfig'] = ('isDisplay', 'squarePolarity', 'sweepIndex', 'indicatorColor')
        log['stimulation']['sweepConfig'] = ('orientation', 'sweepStartCoordinate', 'sweepEndCoordinate')

        return mov, log


class ObliqueKSstimAllDir(object):
    """
    generate Kalatsky & Stryker stimulation in all four direction contiuously
    """
    def __init__(self,
                 monitor,
                 indicator,
                 background=0., #back ground color [-1,1]
                 coordinate='degree', #'degree' or 'linear'
                 squareSize=25, #size of flickering square
                 squareCenter=(0,0), #coordinate of center point
                 flickerFrame=6,
                 sweepWidth=20., # width of sweeps (unit same as Map, cm or deg)
                 stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                 sweepFrame=1,
                 iteration=1,
                 preGapDur=2.,
                 postGapDur=3.,
                 rotation_angle=np.pi/4):

        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate
        self.squareSize = squareSize
        self.squareCenter = squareCenter
        self.flickerFrame = flickerFrame
        self.sweepWidth = sweepWidth
        self.stepWidth = stepWidth
        self.sweepFrame = sweepFrame
        self.iteration = iteration
        self.preGapDur = preGapDur
        self.postGapDur = postGapDur
        self.rotation_angle = rotation_angle


    def generate_movie(self):

        KS_stim=ObliqueKSstim(self.monitor,
                              self.indicator,
                              background=self.background,
                              coordinate=self.coordinate,
                              direction='B2U',
                              squareSize=self.squareSize,
                              squareCenter=self.squareCenter,
                              flickerFrame=self.flickerFrame,
                              sweepWidth=self.sweepWidth,
                              stepWidth=self.stepWidth,
                              sweepFrame=self.sweepFrame,
                              iteration=self.iteration,
                              preGapDur=self.preGapDur,
                              postGapDur=self.postGapDur,
                              rotation_angle=self.rotation_angle)

        movB2U, dictB2U = KS_stim.generate_movie()
        KS_stim.set_direction('U2B')
        movU2B, dictU2B = KS_stim.generate_movie()
        KS_stim.set_direction('L2R')
        movL2R, dictL2R = KS_stim.generate_movie()
        KS_stim.set_direction('R2L')
        movR2L, dictR2L = KS_stim.generate_movie()

        mov = np.vstack((movB2U,movU2B,movL2R,movR2L))
        log = {'monitor':dictB2U['monitor'],
               'indicator':dictB2U['indicator']}
        stimulation = dict(dictB2U['stimulation'])
        stimulation['stimName'] = 'ObliqueKSstimAllDir'
        stimulation['direction'] = ['B2U','U2B','L2R','R2L']

        sweepTable = []
        frames = []

        sweepTableB2U = dictB2U['stimulation']['sweepTable']; framesB2U = dictB2U['stimulation']['frames']; sweepLenB2U = len(sweepTableB2U)
        sweepTableB2U = [ ['B2U', x[1], x[2]] for x in sweepTableB2U]; framesB2U = [[x[0],x[1],x[2],x[3],'B2U'] for x in framesB2U]
        sweepTable += sweepTableB2U; frames += framesB2U

        sweepTableU2B = dictU2B['stimulation']['sweepTable']; framesU2B = dictU2B['stimulation']['frames']; sweepLenU2B = len(sweepTableU2B)
        sweepTableU2B = [ ['U2B', x[1], x[2]] for x in sweepTableU2B]; framesU2B = [[x[0],x[1],x[2],x[3],'U2B'] for x in framesU2B]
        for frame in framesU2B:
            if frame[2] is not None: frame[2] += sweepLenB2U
        sweepTable += sweepTableU2B; frames += framesU2B

        sweepTableL2R = dictL2R['stimulation']['sweepTable']; framesL2R = dictL2R['stimulation']['frames']; sweepLenL2R = len(sweepTableL2R)
        sweepTableL2R = [ ['L2R', x[1], x[2]] for x in sweepTableL2R]; framesL2R = [[x[0],x[1],x[2],x[3],'L2R'] for x in framesL2R]
        for frame in framesL2R:
            if frame[2] is not None: frame[2] += sweepLenB2U+sweepLenU2B
        sweepTable += sweepTableL2R; frames += framesL2R

        sweepTableR2L = dictR2L['stimulation']['sweepTable']; framesR2L = dictR2L['stimulation']['frames']
        sweepTableR2L = [ ['R2L', x[1], x[2]] for x in sweepTableR2L]; framesR2L = [[x[0],x[1],x[2],x[3],'R2L'] for x in framesR2L]
        for frame in framesR2L:
            if frame[2] is not None: frame[2] += sweepLenB2U+sweepLenU2B+sweepLenL2R
        sweepTable += sweepTableR2L; frames += framesR2L

        stimulation['frames'] = [tuple(x) for x in frames]
        stimulation['sweepTable'] = [tuple(x) for x in sweepTable]

        log['stimulation'] = stimulation

        return mov, log

         
class DisplaySequence(object):
    """
    Display the numpy sequence from memory
    """        
    
    def __init__(self,
                 logdir,
                 backupdir=None,
                 displayIteration=1,
                 displayOrder=1,  # 1: the right order; -1: the reverse order
                 mouseid='Test',
                 userid='Jun',
                 psychopyMonitor='testMonitor',
                 isInterpolate=False,
                 isRemoteSync=False,
                 remoteSyncIP='localhost',
                 remoteSyncPort=10003,
                 remoteSyncTriggerEvent="PositiveEdge",
                 remoteSyncSaveWaitTime=3.,
                 isVideoRecord=False,
                 isTriggered=True,
                 triggerNIDev='Dev1',
                 triggerNIPort=1,
                 triggerNILine=0,
                 isSyncPulse=True,
                 syncPulseNIDev='Dev1',
                 syncPulseNIPort=1,
                 syncPulseNILine=1,
                 displayTriggerEvent="NegativeEdge",  # should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"
                 displayScreen=0,
                 initialBackgroundColor=0,
                 videoRecordIP='localhost',
                 videoRecordPort=10000,
                 displayControlIP='localhost',
                 displayControlPort=10002,
                 fileNumNIDev='Dev1',
                 fileNumNIPort='0',
                 fileNumNILines='0:7'):
                     
        self.sequence = None
        self.sequenceLog = {}
        self.psychopyMonitor = psychopyMonitor
        self.remoteSyncSaveWaitTime = remoteSyncSaveWaitTime
        self.isInterpolate = isInterpolate
        self.isRemoteSync = isRemoteSync
        self.remoteSyncIP = remoteSyncIP
        self.remoteSyncPort = remoteSyncPort
        self.remoteSyncTriggerEvent = remoteSyncTriggerEvent
        self.isVideoRecord = isVideoRecord 
        self.isTriggered = isTriggered
        self.triggerNIDev = triggerNIDev
        self.triggerNIPort = triggerNIPort
        self.triggerNILine = triggerNILine
        self.displayTriggerEvent = displayTriggerEvent
        self.isSyncPulse = isSyncPulse
        self.syncPulseNIDev = syncPulseNIDev
        self.syncPulseNIPort = syncPulseNIPort
        self.syncPulseNILine = syncPulseNILine
        self.displayScreen = displayScreen
        self.initialBackgroundColor = initialBackgroundColor
        self.videoRecordIP = videoRecordIP
        self.videoRecordPort = videoRecordPort
        self.displayControlIP = displayControlIP
        self.displayControlPort = displayControlPort
        self.keepDisplay = None
        self.fileNumNIDev = fileNumNIDev
        self.fileNumNIPort = fileNumNIPort
        self.fileNumNILines = fileNumNILines

        try:
            self._remote_obj = RemoteObject(rep_port=self.displayControlPort)
            self._remote_obj.close = self.flag_to_close()
        except Exception as e:
            print e

        # set up remote zro object for sync program
        if self.isRemoteSync:
            self.remoteSync = Proxy(str(self.remoteSyncIP) + ':' + str(self.remoteSyncPort))
        
        if displayIteration % 1 == 0:
            self.displayIteration = displayIteration
        else:
            raise ArithmeticError, "displayIteration should be a whole number."
            
        self.displayOrder = displayOrder
        self.logdir = logdir
        self.backupdir = backupdir
        self.mouseid = mouseid
        self.userid = userid
        self.sequenceLog = None
        
        #FROM DW, setup socket
        try:
            self.displayControlSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.displayControlSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.displayControlSock.bind((self.displayControlIP, self.displayControlPort))

            self.displayControlSock.settimeout(0.0)
        except Exception:
            self.displayControlSock = None
        
        self.clear()


    def set_any_array(self, anyArray, logDict = None):
        """
        to display any numpy 3-d array.
        """
        if len(anyArray.shape) != 3:
            raise LookupError, "Input numpy array should have dimension of 3!"
        
        Vmax = np.amax(anyArray).astype(np.float32)
        Vmin = np.amin(anyArray).astype(np.float32)
        Vrange = (Vmax-Vmin)
        anyArrayNor = ((anyArray-Vmin)/Vrange).astype(np.float16)
        self.sequence = 2*(anyArrayNor-0.5)

        if logDict != None:
            if type(logDict) is dict:
                self.sequenceLog = logDict
            else:
                raise ValueError, '"logDict" should be a dictionary!'
        else:
            self.sequenceLog = {}
        self.clear()
    

    def set_stim(self, stim):
        """
        to display defined stim object
        """
        self.sequence, self.sequenceLog = stim.generate_movie()
        self.clear()


    def trigger_display(self):


        # --------------------------- early preparation for display-----------------------------------------------------
        #test monitor resolution
        try: resolution = self.sequenceLog['monitor']['resolution'][::-1]
        except KeyError: resolution = (800,600)

        try:
            refreshRate = self.sequenceLog['monitor']['refreshRate']
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz.\n"
            refreshRate = 60.

        #prepare display frames log
        if self.sequence is None:
            raise LookupError, "Please set the sequence to be displayed!!\n"
        try:
            sequenceFrames = self.sequenceLog['stimulation']['frames']
            if self.displayOrder == -1: sequenceFrames = sequenceFrames[::-1]
            # generate display Frames
            self.displayFrames=[]
            for i in range(self.displayIteration):
                self.displayFrames += sequenceFrames
        except Exception as e:
            print e
            print "No frame information in sequenceLog dictionary. \nSetting displayFrames to 'None'.\n"
            self.displayFrames = None

        # calculate expected display time
        displayTime = float(self.sequence.shape[0]) * self.displayIteration / refreshRate
        print '\n Expected display time: ', displayTime, ' seconds\n'

        # generate file name
        self._get_file_name()
        print 'File name:', self.fileName + '\n'
        # ---------------------------- early preparation for display----------------------------------------------------


        # ---------------------------setup necessary communication link-------------------------------------------------
        #set up sock communication with video monitoring computer
        if self.isVideoRecord:
            videoRecordSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # set remote sync local path
        # if self.isRemoteSync:
        #     self.remoteSync.set_output_path(os.path.join("c:/sync/output", self.fileName + '-sync.h5'),
        #                                     timestamp=False)

        # ----------------------------setup necessary communication link------------------------------------------------


        # ----------------------------setup psychopy window and stimulus------------------------------------------------
        # start psychopy window
        window = visual.Window(size=resolution,monitor=self.psychopyMonitor,fullscr=True,screen=self.displayScreen,
                               color=self.initialBackgroundColor)
        stim = visual.ImageStim(window, size=(2,2), interpolate=self.isInterpolate)
        # ----------------------------setup psychopy window and stimulus------------------------------------------------


        # initialize keepDisplay
        self.keepDisplay = True

        # handle remote sync start
        if self.isRemoteSync:
            if self.isTriggered:
                syncWait = self._wait_for_trigger(event=self.remoteSyncTriggerEvent)
                if not syncWait:
                    window.close()
                    self.clear()
                    return None
            else:
                pass

            try:
                self._get_file_name()
                print 'File name:', self.fileName + '\n'
                self.remoteSync.set_output_path(os.path.join("c:/sync/output", self.fileName + '-sync.h5'),
                                                timestamp=False)
                self.remoteSync.start()
            except Exception as err:
                print "remote sync object is not started correctly. \n" + str(err) + "\n\n"

        # handle display trigger
        if self.isTriggered:
            displayWait = self._wait_for_trigger(event=self.displayTriggerEvent)
            if not displayWait:
                try:
                    self.remoteSync.stop()
                except Exception:
                    pass
                window.close()
                self.clear()
                return None
            else:
                time.sleep(5.) # wait remote object to start

        # handle video monitoring trigger start
        if self.isVideoRecord:
            videoRecordSock.sendto("1" + self.fileName, (self.videoRecordIP, self.videoRecordPort))  # start eyetracker

        # display
        self._display(window, stim)  # display sequence

        # handle video monitoring trigger stop
        if self.isVideoRecord:
            videoRecordSock.sendto("0"+self.fileName,(self.videoRecordIP,self.videoRecordPort)) #end eyetracker

        # handle remote sync stop
        if self.isRemoteSync:
            syncWait = self._wait_for_trigger(event=self.remoteSyncTriggerEvent)
            if not syncWait:
                try: 
                    self.remoteSync.stop()
                except Exception as err:
                    print "remote sync object is not stopped correctly. \n" + str(err)
                    
                # backup remote sync file 
                try:
                    backupFileFolder = self._get_backup_folder()
                    print '\nRemote sync backup file folder: ' + backupFileFolder + '\n'
                    if backupFileFolder is not None:
                        if not (os.path.isdir(backupFileFolder)): os.makedirs(backupFileFolder)
                        backupFilePath = os.path.join(backupFileFolder,self.fileName+'-sync.h5')
                        time.sleep(self.remoteSyncSaveWaitTime ) # wait remote sync to finish saving
                        self.remoteSync.copy_last_dataset(backupFilePath)
                        print "remote sync dataset saved successfully."
                    else:
                        print "did not find backup path, no remote sync dataset has been saved."
                except Exception as e:
                    print "remote sync dataset is not saved successfully!\n", e
                
                # save display log
                self.save_log()
                                
                # analyze frames
                try:
                    self.frameDuration, self.frame_stats = analyze_frames(ts=self.timeStamp,
                                                                          refreshRate=self.sequenceLog['monitor'][
                                                                              'refreshRate'])
                except KeyError:
                    print "No monitor refresh rate information, assuming 60Hz."
                    self.frameDuration, self.frame_stats = analyze_frames(ts=self.timeStamp, refreshRate=60.)
                self.clear()
                return None
            try:
                self.remoteSync.stop()
            except Exception as err:
                print "remote sync object is not stopped correctly. \n" + str(err)

        self.save_log()

        #analyze frames
        try: self.frameDuration, self.frame_stats = analyze_frames(ts = self.timeStamp, refreshRate = self.sequenceLog['monitor']['refreshRate'])
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz."
            self.frameDuration, self.frame_stats = analyze_frames(ts = self.timeStamp, refreshRate = 60.)

        # backup remote dataset
        if self.isRemoteSync:
            try:
                backupFileFolder = self._get_backup_folder()
                print '\nRemote sync backup file folder: ' + backupFileFolder + '\n'
                if backupFileFolder is not None:
                    if not (os.path.isdir(backupFileFolder)): os.makedirs(backupFileFolder)
                    backupFilePath = os.path.join(backupFileFolder,self.fileName+'-sync.h5')
                    time.sleep(self.remoteSyncSaveWaitTime )  # wait remote sync to finish saving
                    self.remoteSync.copy_last_dataset(backupFilePath)
                    print "remote sync dataset saved successfully."
                else:
                    print "did not find backup path, no remote sync dataset has been saved."
            except Exception as e:
                print "remote sync dataset is not saved successfully!\n", e

        #clear display data
        self.clear()


    def _wait_for_trigger(self, event):
        """
        time place holder for waiting for trigger

        event should be: 'LowLevel', 'HighLevel', 'NegativeEdge' or 'PositiveEdge'

        return True if trigger is detected
               False if manual stop signal is detected
        """

        #check NI signal
        triggerTask = iodaq.DigitalInput(self.triggerNIDev, self.triggerNIPort, self.triggerNILine)
        triggerTask.StartTask()

        print "Waiting for trigger: " + event + ' on ' + triggerTask.devstr

        if event == 'LowLevel':
            lastTTL = triggerTask.read()
            while lastTTL != 0 and self.keepDisplay:
                lastTTL = triggerTask.read()[0]
                self._update_display_status()
            else:
                if self.keepDisplay: triggerTask.StopTask(); print 'Trigger detected. Start displaying...\n\n'; return True
                else: triggerTask.StopTask(); print 'Manual stop signal detected during waiting period. Stop the program.'; return False
        elif event == 'HighLevel':
            lastTTL = triggerTask.read()[0]
            while lastTTL != 1 and self.keepDisplay:
                lastTTL = triggerTask.read()[0]
                self._update_display_status()
            else:
                if self.keepDisplay: triggerTask.StopTask(); print 'Trigger detected. Start displaying...\n\n'; return True
                else: triggerTask.StopTask(); print 'Manual stop signal detected during waiting period. Stop the program.'; return False
        elif event == 'NegativeEdge':
            lastTTL = triggerTask.read()[0]
            while self.keepDisplay:
                currentTTL = triggerTask.read()[0]
                if (lastTTL == 1) and (currentTTL == 0):break
                else:lastTTL = int(currentTTL);self._update_display_status()
            else: triggerTask.StopTask(); print 'Manual stop signal detected during waiting period. Stop the program.';return False
            triggerTask.StopTask(); print 'Trigger detected. Start displaying...\n\n'; return True
        elif event == 'PositiveEdge':
            lastTTL = triggerTask.read()[0]
            while self.keepDisplay:
                currentTTL = triggerTask.read()[0]
                if (lastTTL == 0) and (currentTTL == 1):break
                else:lastTTL = int(currentTTL);self._update_display_status()
            else: triggerTask.StopTask(); print 'Manual stop signal detected during waiting period. Stop the program.'; return False
            triggerTask.StopTask(); print 'Trigger detected. Start displaying...\n\n';  return True
        else:raise NameError, 'trigger should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"!'


    def _get_file_name(self):
        """
        generate the file name of log file
        """

        try:
            self.fileName = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + \
                            self.sequenceLog['stimulation']['stimName'] + \
                            '-M' + \
                            self.mouseid + \
                            '-' + \
                            self.userid
        except KeyError:
            self.fileName = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + 'customStim' + '-M' + self.mouseid + '-' + \
                            self.userid
        
        fileNumber = self._get_file_number()
        
        if self.isTriggered: self.fileName += '-' + str(fileNumber)+'-Triggered'
        else: self.fileName += '-' + str(fileNumber) + '-notTriggered'


    def _get_file_number(self):
        """
        get synced file number for log file name
        """
        
        try:
            fileNumTask = iodaq.DigitalInput(self.fileNumNIDev,self.fileNumNIPort,self.fileNumNILines)
            fileNumTask.StartTask()
            array = fileNumTask.read()
            numStr = (''.join([str(line) for line in array]))[::-1]
            fileNumber = int(numStr, 2)
            # print array, fileNumber
        except Exception as e:
            print e
            fileNumber = None

        return fileNumber
        

    def _display(self, window, stim):
        
        
        # display frames
        timeStamp=[]
        startTime = time.clock()
        singleRunFrames = self.sequence.shape[0]
        
        if self.isSyncPulse:
            syncPulseTask = iodaq.DigitalOutput(self.syncPulseNIDev, self.syncPulseNIPort, self.syncPulseNILine)
            syncPulseTask.StartTask()
            _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

        i = 0

        while self.keepDisplay and i < (singleRunFrames * self.displayIteration):

            if self.displayOrder == 1:frameNum = i % singleRunFrames

            if self.displayOrder == -1:frameNum = singleRunFrames - (i % singleRunFrames) -1

            # currFrame=Image.fromarray(self.sequence[frameNum]) # removed PIL dependency
            stim.setImage(self.sequence[frameNum][::-1,:])
            stim.draw()
            timeStamp.append(time.clock()-startTime)

            #set syncPuls signal
            if self.isSyncPulse: _ = syncPulseTask.write(np.array([1]).astype(np.uint8))
            # print syncPulseTask.readLines()
            #show visual stim
            window.flip()
            #set syncPuls signal
            if self.isSyncPulse: _ = syncPulseTask.write(np.array([0]).astype(np.uint8))
            # print syncPulseTask.readLines()

            self._update_display_status()
            i=i+1
            
        # timeStamp.append(time.clock()-startTime)
        stopTime = time.clock()
        window.close()
        
        if self.isSyncPulse:syncPulseTask.StopTask()
        
        self.timeStamp = np.array(timeStamp)
        self.displayLength = stopTime-startTime

        if self.displayFrames is not None:
            self.displayFrames = self.displayFrames[:i]

        if self.keepDisplay == True: print '\nDisplay successfully completed.'


    def flag_to_close(self):
        self.keepDisplay = False


    def _update_display_status(self):

        if self.keepDisplay is None: raise LookupError, 'self.keepDisplay should start as True for updating display status'

        #check keyboard input 'q' or 'escape'
        keyList = event.getKeys(['q','escape'])
        if len(keyList) > 0:
            self.keepDisplay = False
            print "Keyboard stop signal detected. Stop displaying. \n"

        try:
            msg, addr =  self.displayControlSock.recvfrom(128)
            if msg[0:4].upper() == 'STOP':
                self.keepDisplay = False
                print "Remote stop signal detected. Stop displaying. \n"
        except: pass
    
        if self.isRemoteSync:
            self._remote_obj._check_rep()
    

    def set_display_order(self, displayOrder):
        
        self.displayOrder = displayOrder
        self.clear()
    

    def set_display_iteration(self, displayIteration):
        
        if displayIteration % 1 == 0:self.displayIteration = displayIteration
        else:raise ArithmeticError, "displayIteration should be a whole number."
        self.clear()
        

    def save_log(self):
        
        if self.displayLength is None:
            self.clear()
            raise LookupError, "Please display sequence first!"

        if self.fileName is None:
            self._get_file_name()
            
        if self.keepDisplay == True:
            self.fileName += '-complete'
        elif self.keepDisplay == False:
            self.fileName += '-incomplete'

        #set up log object
        directory = self.logdir + '\sequence_display_log'
        if not(os.path.isdir(directory)):os.makedirs(directory)
        
        logFile = dict(self.sequenceLog)
        displayLog = dict(self.__dict__)
        if hasattr(self, '_remote_obj'):
            displayLog.pop("_remote_obj")
        displayLog.pop('sequenceLog')
        displayLog.pop('displayControlSock')
        displayLog.pop('sequence')
        if hasattr(self, 'remoteSync'):
            displayLog.pop("remoteSync")
        logFile.update({'presentation':displayLog})

        filename =  self.fileName + ".pkl"
        
        #generate full log dictionary
        path = os.path.join(directory, filename)
        ft.saveFile(path,logFile)
        print ".pkl file generated successfully."
        
        
        backupFileFolder = self._get_backup_folder()
        if backupFileFolder is not None:
            if not (os.path.isdir(backupFileFolder)): os.makedirs(backupFileFolder)
            backupFilePath = os.path.join(backupFileFolder,filename)
            ft.saveFile(backupFilePath,logFile)
            print ".pkl backup file generate successfully"
        else:
            print "did not find backup path, no backup has been saved."
            
            
    def _get_backup_folder(self):
        
        if self.fileName is None:
            raise LookupError, 'self.fileName not found.'
        else:
        
            if self.backupdir is not None:

                currDate = self.fileName[0:6]
                stimName = self.sequenceLog['stimulation']['stimName']
                if 'KSstim' in stimName:
                    backupFileFolder = os.path.join(self.backupdir,currDate+'-M'+self.mouseid+'-Retinotopy')
                else:
                    backupFileFolder = os.path.join(self.backupdir,currDate+'-M'+self.mouseid+'-'+stimName)
                return backupFileFolder
            else:
                return None


    def clear(self):
        """ clear display information. """
        self.displayLength = None
        self.timeStamp = None
        self.frameDuration = None
        self.displayFrames = None
        self.frame_stats = None
        self.fileName = None
        self.keepDisplay = None


if __name__ == "__main__":

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=5)
    # indicator=Indicator(mon)
    # KS_stim=KSstim(mon,indicator)
    # displayIteration = 2
    # # print (len(KSstim.generate_frames())*displayIteration)/float(mon.refreshRate)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=r'C:\data',isTriggered=True,displayIteration=2)
    # ds.set_stim(KS_stim)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=20)
    # indicator=Indicator(mon)
    # noise_KS_stim=NoiseKSstim(mon,indicator)
    # displayIteration = 2
    # # print (len(NoiseKSstim.generate_frames())*displayIteration)/float(mon.refreshRate)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=r'C:\data',isTriggered=True,displayIteration=2)
    # ds.set_stim(noise_KS_stim)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=20)
    # indicator=Indicator(mon)
    # flash_noise=FlashingNoise(mon,indicator)
    # displayIteration = 2
    # # print (len(NoiseKSstim.generate_frames())*displayIteration)/float(mon.refreshRate)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=r'C:\data',isTriggered=True,displayIteration=2)
    # ds.set_stim(flash_noise)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=10)
    # indicator=Indicator(mon)
    # gaussian_noise=GaussianNoise(mon,indicator,isWarp=True,enhanceExp=0.5)
    # displayIteration = 2
    # # print (len(NoiseKSstim.generate_frames())*displayIteration)/float(mon.refreshRate)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=r'C:\data',isTriggered=True,displayIteration=2)
    # ds.set_stim(gaussian_noise)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=10)
    # indicator=Indicator(mon)
    # flashing_circle=FlashingCircle(mon,indicator)
    # displayIteration = 2
    # print (len(flashing_circle.generate_frames())*displayIteration)/float(mon.refreshRate)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=r'C:\data',isTriggered=True,displayIteration=2)
    # ds.set_stim(flashing_circle)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=Indicator(mon)
    # sparse_noise=SparseNoise(mon,indicator, subregion=(-20.,20.,40.,60.), gridSpace=(10, 10))
    # gridPoints = sparse_noise._generate_grid_points_sequence()
    # gridLocations = np.array([l[0] for l in gridPoints])
    # plt.plot(monitorPoints[:,0],monitorPoints[:,1],'or',mec='#ff0000',mfc='none')
    # plt.plot(gridLocations[:,0], gridLocations[:,1],'.k')
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=Indicator(mon)
    # sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
    # gridPoints = sparse_noise._generate_grid_points_sequence()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=20)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=Indicator(mon)
    # sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
    # sparse_noise.generate_frames()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon = Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    # frame = get_warped_square(mon.degCorX,mon.degCorY,(20.,25.),4.,4.,0.,foregroundColor=1,backgroundColor=0)
    # plt.imshow(frame,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    # monitorPoints = np.transpose(np.array([mon.degCorX.flatten(),mon.degCorY.flatten()]))
    # indicator=Indicator(mon)
    # sparse_noise=SparseNoise(mon,indicator)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,isTriggered=False,isSyncPulse=False,isVideoRecord=False)
    # ds.set_stim(sparse_noise)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=20)
    # indicator=Indicator(mon)
    # KS_stim_all_dir=KSstimAllDir(mon,indicator,stepWidth=0.3)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,displayIteration = 2,isTriggered=False,isSyncPulse=False)
    # ds.set_stim(KS_stim_all_dir)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    # indicator=Indicator(mon)
    # oblique_KS = ObliqueKSstim(mon,indicator,direction='R2L')
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,displayIteration = 2,isTriggered=False,isSyncPulse=False)
    # ds.set_stim(oblique_KS)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1200, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=10)
    # indicator=Indicator(mon)
    # KS_stim_all_dir=ObliqueKSstimAllDir(mon,indicator,stepWidth=0.15,rotation_angle=np.pi/4)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,displayIteration = 2,isTriggered=False,isSyncPulse=False,isInterpolate=True)
    # ds.set_stim(KS_stim_all_dir)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
    # mon=Monitor(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=16.22,downSampleRate=5)
    # indicator=Indicator(mon)
    #
    # grating = get_grating(mon.degCorX, mon.degCorY, ori=0., spatial_freq=0.1, center=(60.,0.), contrast=1)
    # print grating.max()
    # print grating.min()
    # plt.imshow(grating,cmap='gray',interpolation='nearest',vmin=0., vmax=1.)
    # plt.show()
    #
    # drifting_grating = DriftingGratingCircle(mon,indicator, sf_list=(0.08,0.16),
    #                                          tf_list=(4.,8.), dire_list=(0.,0.1),
    #                                          con_list=(0.5,1.), size_list=(5.,10.),)
    # print '\n'.join([str(cond) for cond in drifting_grating._generate_all_conditions()])
    #
    # drifting_grating2 = DriftingGratingCircle(mon,indicator,
    #                                           center=(60.,0.),
    #                                           sf_list=[0.08, 0.16],
    #                                           tf_list=[4.,2.],
    #                                           dire_list=[np.pi/6],
    #                                           con_list=[1.,0.5],
    #                                           size_list=[40.],
    #                                           blockDur=2.,
    #                                           preGapDur=2.,
    #                                           postGapDur=3.,
    #                                           midGapDur=1.)
    # frames =  drifting_grating2.generate_frames()
    # print '\n'.join([str(frame) for frame in frames])
    #
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,displayIteration = 2,isTriggered=False,isSyncPulse=False,isInterpolate=False)
    # ds.set_stim(drifting_grating2)
    # ds.trigger_display()
    # plt.show()
    #==============================================================================================================================

    # ==============================================================================================================================
    # mon=Monitor(resolution=(1200, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30,downSampleRate=5)
    # indicator=Indicator(mon)
    # uniform_contrast = UniformContrast(mon,indicator, duration=10., color=0.)
    # ds=DisplaySequence(logdir=r'C:\data',backupdir=None,displayIteration=2,isTriggered=False,isSyncPulse=False)
    # ds.set_stim(uniform_contrast)
    # ds.trigger_display()
    # plt.show()
    # ==============================================================================================================================

    # ==============================================================================================================================
    mon = Monitor(resolution=(1080, 1920), dis=13.5, monWcm=88.8, monHcm=50.1, C2Tcm=33.1, C2Acm=46.4, monTilt=16.22,
                  downSampleRate=5)
    indicator = Indicator(mon)
    drifting_grating2 = DriftingGratingCircle(mon, indicator,
                                              center=(60., 0.),
                                              sf_list=[0.08],
                                              tf_list=[4.],
                                              dire_list=np.arange(0, 2 * np.pi, np.pi / 4),
                                              con_list=[1.],
                                              size_list=[20.],
                                              blockDur=2.,
                                              preGapDur=2.,
                                              postGapDur=3.,
                                              midGapDur=1.)

    ds = DisplaySequence(logdir=r'C:\data', backupdir=None, displayIteration=1, isTriggered=True, isSyncPulse=True,
                         isInterpolate=False)
    ds.set_stim(drifting_grating2)
    ds.trigger_display()

    # phases = drifting_grating2._generate_phase_list(4.)
    # print phases
    # ==============================================================================================================================

    print 'for debug...'