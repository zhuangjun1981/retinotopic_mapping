# -*- coding: utf-8 -*-
"""
@author: junz


Visual Stimulus codebase implements several classes to interact with 
"""

from psychopy import visual, event
import os
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle
import tifffile as tiff
import core.FileTools as ft
import core.ImageAnalysis as ia
import IO.nidaq as iodaq


def analyze_frames(ts, refresh_rate, check_point=(0.02, 0.033, 0.05, 0.1)):
    """
    Analyze frame durations of time stamp data. input is the time stamps 
    of each frame and the refresh rate of the monitor
    
    Parameters
    ----------
    ts : ndarray
        list of time stamps of each frame measured in seconds
    refresh_rate : float
        the refresh rate of imaging monitor measured in Hz    
    check_point : tuple, optional
        
        
    Returns
    -------
    frame_duration : ndarray
        list containing the length of each time stamp  
    frame_stats : str
        string containing a statistical analysis of the image frames
    """
    
    frame_duration = ts[1::] - ts[0:-1]
    plt.figure()
    plt.hist(frame_duration, bins=np.linspace(0.0, 0.05, num=51))
    refresh_rate = float(refresh_rate)
    
    num_frames = len(ts)-1
    disp_true = ts[-1]-ts[0]
    disp_expect = (len(ts)-1)/refresh_rate
    avg_frame_time = np.mean(frame_duration)*1000
    sdev_frame_time = np.std(frame_duration)*1000
    short_frame = min(frame_duration)*1000
    short_frame_ind = np.nonzero(frame_duration==np.min(frame_duration))[0][0]
    long_frame = max(frame_duration)*1000
    long_frame_ind = np.nonzero(frame_duration==np.max(frame_duration))[0][0]
    
    
    frame_stats = '\n'
    frame_stats += 'Total number of frames    : %d. \n' % num_frames
    frame_stats += 'Total length of display   : %.5f second. \n' % disp_true
    frame_stats += 'Expected length of display: %.5f second. \n' % disp_expect
    frame_stats += 'Mean of frame durations   : %.2f ms. \n' % avg_frame_time
    frame_stats += 'Standard deviation of frame : %.2f ms.\n' % sdev_frame_time
    frame_stats += 'Shortest frame: %.2f ms, index: %d. \n' % (short_frame, 
                                                               short_frame_ind)
    frame_stats += 'Longest frame : %.2f ms, index: %d. \n' % (long_frame, 
                                                               long_frame_ind)
    
    for i in range(len(check_point)):
        check_number = check_point[i]
        frame_number = len(frame_duration[frame_duration>check_number])
        frame_stats += 'Number of frames longer than %d ms: %d; %.2f%% \n' \
                       % (round(check_number*1000), 
                          frame_number, 
                          round(frame_number*10000/(len(ts)-1))/100)
    
    print frame_stats
    
    return frame_duration, frame_stats


def in_hull(p, hull):
    """
    Determine if points in `p` are in `hull`

    `p` should be a `NxK` coordinate matrix of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    
    Parameters
    ----------
    p : array
        NxK coordinate matrix of N points in K dimensions  
    hull : 
        either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay 
        triangulation will be computed
        
    Returns
    -------
    is_in_hull : ndarray of int
        Indices of simplices containing each point. Points outside the 
        triangulation get the value -1.
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def get_warped_square(deg_coord_x,deg_coord_y,center,width,
                      height,ori,foreground_color=1.,background_color=0.):
    """
    Generate a frame (matrix) with single square defined by `center`, `width`, 
    `height` and orientation in degress visual degree value of each pixel is 
    defined by deg_coord_x, and deg_coord_y
    dtype = np.float32, color space, -1:black, 1:white
    
    Parameters
    ----------
    deg_coord_x : ndarray
        contains 
    deg_coord_y : 
        
    center : tuple
        center of the square
    width : 
         width of the square
    height : 
         height of the square
    ori : 
        angle in degree, should be 0~180
    foreground_color : float, optional
         color of the noise pixels, takes values in [-1,1] and defaults to `1.`
    background_color : float, optional
         color of the background behind the noise pixels, takes values in 
         [-1,1] and defaults to `0.`
    Returns
    -------
    frame : ndarray
         the warped s
    """

    frame = np.ones(deg_coord_x.shape,dtype=np.float32)*background_color

    if ori < 0. or ori > 180.: 
         raise ValueError, 'ori should be between 0 and 180.'

    k1 = np.tan(ori*np.pi/180.)
    k2 = np.tan((ori+90.)*np.pi/180.)

    dis_width = np.abs(((k1*deg_coord_x - deg_coord_y 
                         + center[1] - k1 * center[0]) / np.sqrt(k1**2 +1)))
    dis_height = np.abs(((k2*deg_coord_x - deg_coord_y 
                          + center[1] - k2 * center[0]) / np.sqrt(k2**2 +1)))

    frame[np.logical_and(dis_width<=width/2.,
                         dis_height<=height/2.)] = foreground_color

    return frame


def get_circle_mask(map_x, map_y, center, radius):
    """
    Generate a binary mask of a circle with given `center` and `radius` 
    
    The binary mask is generated on a map with coordinates for each pixel 
    defined by `map_x` and `map_y`

    Parameters
    ----------
    map_x  : ndarray
        x coordinates for each pixel on a map
    map_y  : ndarray
        y coordinates for each pixel on a map
    center : tuple
        coordinates of the center of the binary circle mask
    radius : float
        radius of the binary circle mask
    
    Returns
    -------
    circle_mask : 
        binary circle mask, takes values in [0.,1.]
    """

    if map_x.shape != map_y.shape: 
         raise ValueError, 'map_x and map_y should have same shape!'

    if len(map_x.shape) != 2: 
         raise ValueError, 'map_x and map_y should be 2-d!!'

    circle_mask = np.zeros(map_x.shape, dtype = np.uint8)
    for (i, j), value in  np.ndenumerate(circle_mask):
        x=map_x[i,j]; y=map_y[i,j]
        if ia.distance((x,y),center) <= radius:
            circle_mask[i,j] = 1

    return circle_mask


def get_grating(map_x, map_y, ori=0., spatial_freq=0.1,
                center=(0.,60.), phase=0., contrast=1.):
    """
    Generate a grating frame with defined spatial frequency, center location, 
    phase and contrast

    Parameters
    ----------
    map_x : ndarray
        x coordinates for each pixel on a map
    map_y : ndarray
        y coordinates for each pixel on a map
    ori : float, optional
        orientation angle in degrees, defaults to 0.
    spatial_freq : float, optional
        spatial frequency (cycle per unit), defaults to 0.1
    center : tuple, optional
        center coordinates of circle {x, y}
    phase : float, optional
        defaults to 0.
    contrast : float, optional
        defines contrast. takes values in [0., 1.], defaults to 1.
    
    Returns
    -------
    frame : 
        a frame as floating point 2-d array with grating, value range [0., 1.]
    """

    if map_x.shape != map_y.shape: 
        raise ValueError, 'map_x and map_y should have same shape!'

    if len(map_x.shape) != 2: 
        raise ValueError, 'map_x and map_y should be 2-d!!'

    map_x_h = np.array(map_x, dtype = np.float32)
    map_y_h = np.array(map_y, dtype = np.float32)

    distance = (np.sin(ori) * (map_x_h - center[0]) - 
                          np.cos(ori) * (map_y_h - center[1]))

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
                 mon_width_cm, 
                 mon_height_cm, 
                 C2T_cm, 
                 C2A_cm, 
                 mon_tilt, 
                 visual_field='right',
                 deg_coord_x=None, 
                 deg_coord_y=None, 
                 name='testMonitor', 
                 gamma=None, 
                 gamma_grid=None, 
                 luminance=None,
                 downsample_rate=10, 
                 refresh_rate = 60.):
        """
        Initialize monitor object.
        
        This object contains the relevant data for the monitor used within a 
        given experimental setup. When initialized, the rectangular coordinates
        of the pixels on the monitor are computed and stored as `lin_coord_x`, 
        `lin_coord_y`. The rectangular coordinates are then transformed and 
        warped by calling the `remap` method to populate the `deg_coord_x` and
        `deg_coord_y` attributes.
        
        Parameters
        ----------
        resolution : tuple
            value of the monitor resolution
        dis : float 
             distance from eyeball to monitor (in cm)
        mon_width_cm : float
            width of monitor (in cm)
        mon_height_cm : float
            height of monitor (in cm)
        C2T_cm : float
            distance from gaze center to monitor top
        C2A_cm : float
            distance from gaze center to anterior edge of the monitor
        mon_tilt : float
            angle between mouse body axis and monitor plane, in degrees
        visual_field : str from {'right','left'}, optional
            the eye that is facing the monitor, defaults to 'right'
        deg_coord_x : ndarray, optional
             array of warped x pixel coordinates, defaults to `None`
        deg_coord_y : ndarray, optional
             array of warped y pixel coordinates, defaults to `None`
        name : str, optional
             name of the monitor, defaults to `testMonitor`
        gamma : optional
             for gamma correction, defaults to `None`
        gamma_grid : optional
             for gamme correction, defaults to `None`
        luminance : optional
             monitor luminance, defaults to `None`
        downsample_rate : int, optional
             downsample rate of monitor pixels, defaults to 10
        refresh_rate : float, optional
            the refresh rate of the monitor in Hz, defaults to 60 
        """
                     
        if resolution[0] % downsample_rate != 0 \
                       or resolution[1] % downsample_rate != 0:           
           raise ArithmeticError, 'Resolution pixel numbers are not' \
           'divisible by down sampling rate'
        
        self.resolution = resolution
        self.dis = dis
        self.mon_width_cm = mon_width_cm
        self.mon_height_cm = mon_height_cm
        self.C2T_cm = C2T_cm 
        self.C2A_cm = C2A_cm 
        self.mon_tilt = mon_tilt
        self.visual_field = visual_field
        self.deg_coord_x = deg_coord_x
        self.deg_coord_y = deg_coord_y
        self.name = name
        self.downsample_rate = downsample_rate
        self.gamma = gamma
        self.gamma_grid = gamma_grid
        self.luminance = luminance
        self.refresh_rate = 60
        
        #distance form projection point of the eye to bottom of the monitor
        self.C2B_cm = self.mon_height_cm - self.C2T_cm
        #distance form projection point of the eye to right of the monitor
        self.C2P_cm = self.mon_width_cm - self.C2A_cm
        
        resolution=[0,0]        
        resolution[0]=self.resolution[0]/downsample_rate
        resolution[1]=self.resolution[1]/downsample_rate
        
        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]), 
                                               range(resolution[0]))
        
        if self.visual_field == "left": 
            map_x = np.linspace(self.C2A_cm, -1.0 * self.C2P_cm, resolution[1])
            
        if self.visual_field == "right":
            map_x = np.linspace(-1 * self.C2A_cm, self.C2P_cm, resolution[1])
            
        map_y = np.linspace(self.C2T_cm, -1.0 * self.C2B_cm, resolution[0])
        old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse = False)
        
        self.lin_coord_x=old_map_x
        self.lin_coord_y=old_map_y
        
        self.remap()
        
    def set_gamma(self, gamma, gamma_grid):
        self.gamma = gamma
        self.gamma_grid = gamma_grid
        
    def set_luminance(self, luminance):
        self.luminance = luminance
        
    def set_downsample_rate(self, downsample_rate):
        
        if self.resolution[0] % downsample_rate != 0 \
             or self.resolution[1] % downsample_rate != 0:
           
           raise ArithmeticError, 'Resolution pixel numbers are not' \
           'divisible by down sampling rate'
        
        self.downsample_rate = downsample_rate
        
        resolution = [0,0]        
        resolution[0] = self.resolution[0]/downsample_rate
        resolution[1] = self.resolution[1]/downsample_rate
        
        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]), 
                                               range(resolution[0]))
        
        if self.visual_field == "left": 
            map_x = np.linspace(self.C2A_cm, -1.0 * self.C2P_cm, resolution[1])
            
        if self.visual_field == "right":
            map_x = np.linspace(-1 * self.C2P_cm, self.C2P_cm, resolution[1])
            
        map_y = np.linspace(self.C2T_cm, -1.0 * self.C2B_cm, resolution[0])
        old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse = False)
        
        self.lin_coord_x = old_map_x
        self.lin_coord_y = old_map_y
        
        self.remap()
        
        
    def remap(self):
        """
        warp the linear pixel coordinates and populate the `deg_coord_x` and 
        `deg_coord_y` attributes. 
         
        Function is called immediately as soon as the monitor object is 
        initialized.
        """
        
        resolution = [0,0]        
        resolution[0] = self.resolution[0]/self.downsample_rate
        resolution[1] = self.resolution[1]/self.downsample_rate
        
        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]), 
                                               range(resolution[0]))
        
        new_map_x = np.zeros(resolution,dtype=np.float16)
        new_map_y = np.zeros(resolution,dtype=np.float16)
        
        
        for j in range(resolution[1]):
            new_map_x[:, j] = ((180.0 / np.pi) * 
                               np.arctan(self.lin_coord_x[0, j] / self.dis))
            dis2 = np.sqrt(np.square(self.dis) + 
                           np.square(self.lin_coord_x[0, j])) 
            
            for i in range(resolution[0]):
                new_map_y[i, j] = ((180.0 / np.pi) * 
                                   np.arctan(self.lin_coord_y[i, 0] / dis2))
                
        self.deg_coord_x = new_map_x + 90 - self.mon_tilt
        self.deg_coord_y = new_map_y
        
class Indicator(object):
    """
    flashing indicator for photodiode
    """
    def __init__(self,
                 monitor,
                 width_cm = 3.,
                 height_cm = 3.,
                 position = 'northeast',
                 is_sync = True,
                 freq = 2.):
        """
        Initialize indicator object
       
        Parameters
        ----------
        monitor : monitor object
            The monitor used within the experimental setup
        width_cm : float, optional
            width of the size of the indicator in cm, defaults to `3.`
        height_cm : float, optional
             height of the size of the indicator in cm, defaults to `3.`
        position : str from {'northeast','northwest','southwest','southeast'}
             the placement of the indicator, defaults to 'northeast'
        is_sync : bool, optional
             determines whether the indicator is synchronized with the stimulus,
             defaults to True.
        freq : float, optional
            frequency of photodiode, defaults to `2.`
        """
        
        self.monitor=monitor
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.width_pixel, self.height_pixel = self.get_size_pixel()
        self.position = position
        self.center_width_pixel, self.center_height_pixel = self.get_center()
        self.is_sync = is_sync

        if is_sync == False:
            self.freq = freq #if not synchronized with stimulation, self update frquency of the indicator
            self.frame_num = self.get_frames()
        else:
            self.freq = None
            self.frame_num = None

    def get_size_pixel(self):

        screen_width = (self.monitor.resolution[1] / 
                        self.monitor.downsample_rate)
        screen_height = (self.monitor.resolution[0] / 
                         self.monitor.downsample_rate)

        indicator_width = int((self.width_cm / self.monitor.mon_width_cm ) * 
                              screen_width)
        indicator_height = int((self.height_cm / self.monitor.mon_height_cm ) * 
                               screen_height)

        return indicator_width, indicator_height

    def get_center(self):

        screen_width = (self.monitor.resolution[1] / 
                        self.monitor.downsample_rate)
        screen_height = (self.monitor.resolution[0] / 
                         self.monitor.downsample_rate)

        if self.position == 'northeast':
            center_width = screen_width - self.width_pixel / 2
            center_height = self.height_pixel / 2

        elif self.position == 'northwest':
            center_width = self.width_pixel / 2
            center_height = self.height_pixel / 2

        elif self.position == 'southeast':
            center_width = screen_width - self.width_pixel / 2
            center_height = screen_height - self.height_pixel / 2

        elif self.position == 'southwest':
            center_width = self.width_pixel / 2
            center_height = screen_height - self.height_pixel / 2

        else:
            raise LookupError, '`position` not in {"northeast","southeast","northwest","southwest"}'

        return int(center_width), int(center_height)

    def get_frames(self):

        """
        if not synchronized with stimulation, get frame numbers of each update
        of indicator
        """

        refresh_rate = self.monitor.refresh_rate

        if refresh_rate % self.freq != 0:
            raise ArithmeticError, "`freq` not divisble by monitor ref rate."

        return refresh_rate/self.freq

        
class Stim(object):
    """
    generic class for visual stimulation. parent class for individual 
    stimulus routines.
    
    Methods
    -------
    generate_frames : 
        place-holder function for generating stimulus parameters
    generate_movie : 
        place-holder function for generating particular stimulus routine
    """
    def __init__(self,
                 monitor, 
                 indicator, 
                 background = 0., 
                 coordinate = 'degree', 
                 pregap_dur = 2., 
                 postgap_dur = 3.): 
        """
        Initialize visual stimulus object
       
        Parameters
        ----------
        monitor : monitor object
             the monitor used to display stimulus in the experiment
        indicator : indicator object
             the indicator used during stimulus
        background : float, optional
            background color, takes values in [-1,1]    
        coordinate : str {'degree', 'linear'}, optional
            values for coordinates, defaults to 'degree'
        pregap_dur : float, optional
            duration of gap period before stimulus, measured in seconds
        postgap_dur : float, optional
            duration of gap period after stimulus, measured in seconds
        """
    
    
        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate
        self.pregap_dur = pregap_dur
        self.postgap_dur = postgap_dur

        self.clear()

    @property
    def pregap_frame_num(self):
        return int(self.pregap_dur * self.monitor.refresh_rate)

    @property
    def postgap_frame_num(self):
        return int(self.postgap_dur * self.monitor.refresh_rate)

    def generate_frames(self):
        """
        place holder of function "generate_frames" for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'
        
    def generate_movie(self):
        """
        place holder of function "generate_movie" for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'
        
    def clear(self):
        self.frames = None
    
    def set_pre_gap_dur(self,pregap_dur):
        self.pregap_frame_num = int(self.pregap_dur*self.monitor.refresh_rate)
        self.clear()
        
    def set_post_gap_dur(self,postgap_dur):
        self.postgap_frame_num = int(self.postgap_dur*self.monitor.refresh_rate)
        self.clear()


class UniformContrast(Stim):
    """
    Generate full field uniform luminance for recording spontaneous activity.
    Inherits from Stim.
    
    The full field uniform luminance stimulus presents a fixed background color
    which is normally taken to be grey. 
    """

    def __init__(self, monitor, indicator, duration, color=0., pregap_dur=2., 
                 postgap_dur=3., background=0., coordinate='degree'):
        """
        Initialize UniformContrast object
        
        Parameters
        ----------
        monitor : monitor object
            inherited monitor object from `Stim` class
        indicator : indicator object
            inherited indicator object from `Stim` class
        duration : int
            amount of time (in seconds) the stimulus is presented 
        color : float, optional
            color of the uniform 
        pregap_dur : float, optional
            amount of time (in seconds) before the stimulus is presented
        postgap_dur : float, optional
            amount of time (in seconds) after the stimulus is presented
        background : float, optional
            color during pre and post gap
        """

        super(UniformContrast, self).__init__(monitor=monitor, 
                                              indicator=indicator,
                                              coordinate=coordinate, 
                                              background=background,
                                              pregap_dur=pregap_dur,
                                              postgap_dur=postgap_dur)
        
        self.stim_name = 'UniformContrast'
        self.duration = duration
        self.color = color

    def generate_frames(self):
        """
        generate a tuple of parameters of each frame.

        for each frame:

        first element: gap:0 or display:1
        second element: color of indicator, gap:-1, display:1
        """

        displayframe_num = int(self.duration * self.monitor.refresh_rate)

        frames = [(0, -1)] * self.pregap_frame_num + \
                 [(1, 1.)] * displayframe_num + \
                 [(0, -1)] * self.postgap_frame_num

        return tuple(frames)

    def generate_movie(self):
        """
        generate movie for uniform contrast display for recording of spontaneous 
        activity
        
        Returns
        -------
        full_seq : list
            3-d array of the stimulus to be displayed. elements are of type unit8
        full_dict : dict
            dictionary containing the information of the stimulus
        """

        self.frames = self.generate_frames()

        full_seq = np.zeros((len(self.frames), 
                             self.monitor.deg_coord_x.shape[0], 
                             self.monitor.deg_coord_x.shape[1]),
                             dtype=np.float16)

        indicator_width_min = (self.indicator.center_width_pixel - 
                               (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel + 
                               (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel - 
                                (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel + 
                                (self.indicator.height_pixel / 2))

        background = np.ones((np.size(self.monitor.deg_coord_x, 0), 
                              np.size(self.monitor.deg_coord_x, 1)),
                              dtype=np.float16)*self.background

        display = np.ones((np.size(self.monitor.deg_coord_x, 0), 
                           np.size(self.monitor.deg_coord_x, 1)),
                           dtype=np.float16)*self.background

        if not (self.coordinate == 'degree' or self.coordinate == 'linear'):
            raise LookupError, "`coordinate` value not in {'degree','linear'}"

        for i in range(len(self.frames)):
            curr_frame = self.frames[i]

            if curr_frame[0] == 0:
                curr_FC_seq = background
            else:
                curr_FC_seq = display

            curr_FC_seq[indicator_height_min:indicator_height_max, 
                           indicator_width_min:indicator_width_max] = curr_frame[1]

            full_seq[i] = curr_FC_seq

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print ['Generating numpy sequence: ' + 
                       str(int(100 * (i + 1) / len(self.frames))) + '%']

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        full_dict = {'stimulation': NFdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict


class FlashingCircle(Stim):
    """
    flashing circle stimulus
    """

    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree',
                 center = (90., 10.), 
                 radius = 10.,
                 color = -1., 
                 iteration= 1, 
                 flash_frame= 3,
                 pregap_dur=2., 
                 postgap_dur=3., 
                 background = 0.):
        
        """
        Initialize `FlashingCircle` stimulus object. 
        
        Parameters
        ----------
        stim_name : str
            Name of the stimulus
        center : 2-tuple, optional
            center coordinate of the circle in degrees, defaults to `(90.,10.)`
        radius : float, optional
            radius of the circle, defaults to `10.`
        color : float, optional
            color of the circle, takes values in [-1,1], defaults to `-1.`
        iteration : int, optional
            total number of flashes, defaults to `1`
        flash_frame : int, optional
            frame number that circle is displayed during each flash, defaults 
            to `3`
        """

        super(FlashingCircle,self).__init__(monitor=monitor, 
                                           indicator=indicator, 
                                           background=background,
                                           coordinate=coordinate, 
                                           pregap_dur=pregap_dur, 
                                           postgap_dur=postgap_dur)

        self.stim_name = 'FlashingCircle'
        self.center = center
        self.radius = radius
        self.color = color
        self.iteration = iteration
        self.flash_frame = flash_frame
        self.frame_config = ('is_display', 'is_iteration_start', 
                             'current_iteration', 'indicator_color')

        self.clear()

    def set_flash_frame_num(self, flash_frame_num):
        self.flash_frame = flash_frame_num
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
        function to generate all the frames needed for the stimulation. 
        Returns a list of information of all frames as a list of tuples

        Information contained in each frame:
           first element : 
                during a gap, the value is equal to 0 and during display the
                value is equal to 1
           second element : 
                the value is equal to 1 on the frame that the stimulus begins,
                and is equal to 0 otherwise.
           third element : 
                value corresponds to the current iteration
           fourth element : 
                corresponds to the color of indicator and during stimulus 
                the value is equal to 1, whereas during a gap the value is 
                equal to 0
        """

        #frame number for each iteration
        iteration_frame_num = (self.pregap_frame_num + 
                               self.flash_frame + self.postgap_frame_num)

        frames = np.zeros((self.iteration*(iteration_frame_num),4)).astype(np.int)

        #initilize indicator color
        frames[:,3] = -1

        for i in xrange(frames.shape[0]):

            # current iteration number
            frames[i, 2] = i // iteration_frame_num

            # mark start frame of every iteration
            if i % iteration_frame_num == 0:
                frames[i, 1] = 1

            # mark display frame and synchronized indicator
            if ((i % iteration_frame_num >= self.pregap_frame_num) and \
               (i % iteration_frame_num < (self.pregap_frame_num + self.flash_frame))):

                frames[i, 0] = 1

                if self.indicator.is_sync:
                    frames[i, 3] = 1

            # mark unsynchronized indicator
            if not(self.indicator.is_sync):
                if np.floor(i // self.indicator.frame_num) % 2 == 0:
                    frames[i, 3] = 1
                else:
                    frames[i, 3] = -1

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generate movie frame by frame.
        """

        self.frames = self.generate_frames()

        full_seq = np.zeros((len(self.frames),self.monitor.deg_coord_x.shape[0],
                             self.monitor.deg_coord_x.shape[1]),
                             dtype=np.float16)

        indicator_width_min = (self.indicator.center_width_pixel - 
                               (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel + 
                               (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel - 
                                (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel + 
                                (self.indicator.height_pixel / 2))

        background = np.ones((np.size(self.monitor.deg_coord_x, 0), 
                              np.size(self.monitor.deg_coord_x,1)), 
                              dtype = np.float16)*self.background

        if self.coordinate == 'degree':
            map_x = self.monitor.deg_coord_x
            map_y = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_x = self.monitor.lin_coord_x
            map_y = self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        circle_mask = get_circle_mask(map_x, map_y, 
                                 self.center, self.radius).astype(np.float16)

        for i in range(len(self.frames)):
            curr_frame = self.frames[i]

            if curr_frame[0] == 0:
                curr_FC_seq = background
            else:
                curr_FC_seq = ((circle_mask * self.color) + 
                                  ((-1 * (circle_mask - 1)) * background))

            curr_FC_seq[indicator_height_min:indicator_height_max, 
                           indicator_width_min:indicator_width_max] = curr_frame[3]

            full_seq[i] = curr_FC_seq

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '
                       +str(int(100 * (i+1) / len(self.frames)))+'%']

        mondict=dict(self.monitor.__dict__)
        indicator_dict=dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        full_dict={'stimulation':NFdict,
                   'monitor':mondict,
                   'indicator':indicator_dict}

        return full_seq, full_dict


class SparseNoise(Stim):
    """
    generate sparse noise stimulus integrates flashing indicator for photodiode
    """

    def __init__(self,
                 monitor,
                 indicator,
                 background=0., 
                 coordinate='degree', 
                 grid_space=(10.,10.), 
                 probe_size=(10.,10.), 
                 probe_orientation=0., 
                 probe_frame_num=3, 
                 subregion=None, 
                 sign='ON-OFF', 
                 iteration=1,
                 pregap_dur=2.,
                 postgap_dur=3.):
       
        super(SparseNoise,self).__init__(monitor=monitor,
                                         indicator=indicator,
                                         background=background,
                                         coordinate = coordinate,
                                         pregap_dur=pregap_dur,
                                         postgap_dur=postgap_dur)
        """    
        Initialize sparse noise object, inherits attributes from Stim object
        
        Attributes
        ----------
        stim_name : str
            Name of stimulus
        grid_space : 2-tuple of floats, optional
            first coordinate is altitude, second coordinate is azimuth
        probe_size : 2-tuple of floats, optional
            size of flicker probes. First coordinate defines the width, and
            second coordinate defines the height
        probe_orientation : float, optional
            orientation of flicker probes
        probe_frame_num : int, optional
            number of frames for each square presentation
        subregion : 
            the region on the monitor that will display the sparse noise,
            list or tuple 
        sign : {'ON-OFF', 'ON', 'OFF'}, optional
            
        iteration : int, optional
            number of times to present stimulus, defaults to `1`
        """


        self.stim_name = 'SparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.probe_orientation = probe_orientation
        self.probe_frame_num = probe_frame_num
        self.frame_config = ('is_display', '(azimuth, altitude)', 
                             'polarity', 'indicator_color')

        if subregion is None:
            if self.coordinate == 'degree':
                self.subregion = [np.amin(self.monitor.deg_coord_y), 
                                  np.amax(self.monitor.deg_coord_y),
                                  np.amin(self.monitor.deg_coord_x), 
                                  np.amax(self.monitor.deg_coord_x)]
            if self.coordinate == 'linear':
                self.subregion = [np.amin(self.monitor.lin_coord_y), 
                                  np.amax(self.monitor.lin_coord_y),
                                  np.amin(self.monitor.lin_coord_x), 
                                  np.amax(self.monitor.lin_coord_x)]
        else:
            self.subregion = subregion

        self.sign = sign
        self.iteration = iteration

        self.clear()

    def _getgrid_points(self):
        """
        generate all the grid points in display area s
        [azi, alt]
        """

        rows = np.arange(self.subregion[0], 
                         self.subregion[1] + self.grid_space[0], 
                         self.grid_space[0])
        columns = np.arange(self.subregion[2], 
                            self.subregion[3] + self.grid_space[1], 
                            self.grid_space[1])

        xx,yy = np.meshgrid(columns,rows)

        grid_points = np.transpose(np.array([xx.flatten(),yy.flatten()]))

        #get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            mon_points = np.transpose(np.array([self.monitor.deg_coord_x.flatten(),
                                                self.monitor.deg_coord_y.flatten()]))
        if self.coordinate == 'linear':
            mon_points = np.transpose(np.array([self.monitor.lin_coord_x.flatten(),
                                                self.monitor.lin_coord_y.flatten()]))

        #get the grid points within the coverage of monitor
        grid_points = grid_points[in_hull(grid_points,mon_points)]

        return grid_points

    def _generate_grid_points_sequence(self):
        """
        generate pseudorandomized grid point sequence. if ON-OFF, consecutive 
        frames should not present stimulus at same location
        
        Returns
        -------
        all_grid_points : list 
            list of [grid_point, sign]
        """

        grid_points = self._getgrid_points()

        if self.sign == 'ON':
            grid_points = [[x,1] for x in grid_points]
            shuffle(grid_points)
            return grid_points
        elif self.sign == 'OFF':
            grid_points = [[x,-1] for x in grid_points]
            shuffle(grid_points)
            return grid_points
        elif self.sign == 'ON-OFF':
            all_grid_points = [[x,1] for x in grid_points] + [[x,-1] for x in grid_points]
            shuffle(all_grid_points)
            # remove coincident hit of same location by continuous frames
            print 'removing coincident hit of same location with continuous frames:'
            while True:
                iteration = 0
                coincident_hit_num = 0
                for i, grid_point in enumerate(all_grid_points[:-3]):
                    if (all_grid_points[i][0] == all_grid_points[i+1][0]).all():
                        all_grid_points[i+1], all_grid_points[i+2] = all_grid_points[i+2], all_grid_points[i+1]
                        coincident_hit_num += 1
                iteration += 1
                print 'iteration:',iteration,'  continous hits number:',coincident_hit_num
                if coincident_hit_num == 0:
                    break

            return all_grid_points

    def generate_frames(self):
        """
        function to generate all the frames needed for SparseNoiseStimu

        returns a list of information of all frames as a list of tuples

        Information contained in each frame:
             first element: 
                  when stimulus is displayed value is equal to 1, otherwise
                  equal to 0,
             second element: tuple, 
                  retinotopic location of the center of current square,[azi,alt]
             third element: 
                  polarity of current square, 1 -> bright, -1-> dark
             forth element: color of indicator
                  if synchronized : value equal to 0 when stimulus is not 
                       begin displayed, and 1 for onset frame of stimulus for 
                       each square, -1 for the rest.
                  if non-synchronized: values alternate between -1 and 1 
                       at defined frequency
                       
             for gap frames the second and third elements should be 'None'
        """

        frames = []
        if self.probe_frame_num == 1:
            indicator_on_frame = 1
        elif self.probe_frame_num > 1:
            indicator_on_frame = self.probe_frame_num // 2
        else:
            raise ValueError('`probe_frame_num` should be an int larger than 0!')

        indicator_off_frame = self.probe_frame_num - indicator_on_frame

        for i in range(self.iteration):

            if self.pregap_frame_num>0: 
                 frames += [[0,None,None,-1]]*self.pregap_frame_num

            iter_grid_points = self._generate_grid_points_sequence()

            for grid_point in iter_grid_points:
                frames += [[1,grid_point[0],grid_point[1],1]] * indicator_on_frame
                frames += [[1,grid_point[0],grid_point[1],-1]] * indicator_off_frame

            if self.postgap_frame_num>0: 
                 frames += [[0,None,None,-1]]*self.postgap_frame_num

        if self.indicator.is_sync == False:
            indicator_frame = self.indicator.frame_num
            for m in range(len(frames)):
                if np.floor(m // indicator_frame) % 2 == 0:
                    frames[m][3] = 1
                else:
                    frames[m][3] = -1

        frames = tuple(frames)

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def generate_movie(self):
        """
        generate movie for display frame by frame
        """

        self.frames = self.generate_frames()

        if self.coordinate=='degree':
             coord_x=self.monitor.deg_coord_x
             coord_y=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_x=self.monitor.lin_coord_x
             coord_y=self.monitor.lin_coord_y

        indicator_width_min = (self.indicator.center_width_pixel - 
                         (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel + 
                          (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel - 
                         (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel + 
                         (self.indicator.height_pixel / 2))

        full_seq = np.ones((len(self.frames),
                                self.monitor.deg_coord_x.shape[0],
                                self.monitor.deg_coord_x.shape[1]),
                                 dtype=np.float16) * self.background

        for i, curr_frame in enumerate(self.frames):
            if curr_frame[0] == 1: # not a gap
                if i == 0: # first frame and (not a gap)
                    curr_disp_mat = get_warped_square(coord_x, 
                                                      coord_y,
                                                      center = curr_frame[1], 
                                                      width=self.probe_size[0],
                                                      height=self.probe_size[1], 
                                                      ori=self.probe_orientation,
                                                      foreground_color=curr_frame[2], 
                                                      background_color=self.background)
                else: # (not first frame) and (not a gap)
                    if self.frames[i-1][1] is None: # (not first frame) and (not a gap) and (new square from gap)
                        curr_disp_mat = get_warped_square(coord_x, 
                                                          coord_y, 
                                                          center = curr_frame[1], 
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1], 
                                                          ori=self.probe_orientation,
                                                          foreground_color=curr_frame[2], 
                                                          background_color=self.background)
                    elif (curr_frame[1]!=self.frames[i-1][1]).any() or (curr_frame[2]!=self.frames[i-1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        curr_disp_mat = get_warped_square(coord_x, 
                                                          coord_y, 
                                                          center = curr_frame[1], 
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1], 
                                                          ori=self.probe_orientation,
                                                          foreground_color=curr_frame[2], 
                                                          background_color=self.background)

                #assign current display matrix to full sequence
                full_seq[i] = curr_disp_mat

            #add sync square for photodiode
            full_seq[i, indicator_height_min:indicator_height_max, 
                     indicator_width_min:indicator_width_max] = curr_frame[3]

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+
                       str(int(100 * (i+1) / len(self.frames)))+'%']

        #generate log dictionary
        mondict=dict(self.monitor.__dict__)
        indicator_dict=dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        SNdict=dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        full_dict={'stimulation':SNdict,
                        'monitor':mondict,
                        'indicator':indicator_dict}

        return full_seq, full_dict


class DriftingGratingCircle(Stim):
    """
    class of drifting grating circle stimulus
    """

    def __init__(self,
                 monitor,
                 indicator,
                 background=0., 
                 coordinate='degree',
                 center=(60.,0.), 
                 sf_list=(0.08,),
                 tf_list=(4.,), 
                 dire_list=(0.,),
                 con_list=(0.5,), 
                 size_list=(5.,), 
                 block_dur=2., 
                 midgap_dur=0.5, 
                 iteration=1, 
                 pregap_dur=2.,
                 postgap_dur=3.):
       
        super(DriftingGratingCircle,self).__init__(monitor=monitor,
                                                   indicator=indicator,
                                                   background=background,
                                                   coordinate=coordinate,
                                                   pregap_dur=pregap_dur,
                                                   postgap_dur=postgap_dur)
        """
        Initialize `DriftingGratingCircle` stimulus object, inherits attributes
        from `Stim` class.
        
        Parameters
        ----------
        center : 2-tuple of floats, optional
            coordintes for center of the stimulus (azimuth, argument) 
        sf_list : n-tuple, optional
            list of spatial frequencies in cycles/unit, defaults to `(0.08)`
        tf_list : n-tuple, optional
            list of temportal frequencies in Hz, defaults to `(4.)`
        dire_list : n-tuple, optional
            list of directions in degrees, defaults to `(0.)`
        con_list : n-tuple, optional
            list of contrasts taking values in [0.,1.], defaults to `(0.5)`
        size_list : n-tuple
           list of radii of circles, unit defined by `self.coordinate`, defaults
           to `(5.)`
        block_dur = float, optional
            duration of each condition in seconds, defaults to `2.`
        midgap_dur = float, optional
            duration of gap between conditions, defaults to `0.5`
        iteration = int, optional
            number of times the stimulus is displayed, defaults to `1`
        """
        self.stim_name = 'DriftingGratingCircle'
        self.center = center
        self.sf_list = sf_list
        self.tf_list = tf_list
        self.dire_list = dire_list
        self.con_list = con_list
        self.size_list = size_list
        self.block_dur = float(block_dur)
        self.midgap_dur = float(midgap_dur)
        self.iteration = iteration
        self.frame_config = ('is_display', 'isCycleStart', 'spatialFrequency', 
                            'temporalFrequency', 'direction',
                            'contrast', 'radius', 'phase', 'indicator_color')

        for tf in tf_list:
            period = 1. / tf
            if (0.05 * period) < (block_dur % period) < (0.95 * period):
                print period
                print block_dur % period
                print 0.95 * period
                error_msg = ('Duration of each block times tf '+ str(tf) 
                             + ' should be close to a whole number!')
                raise ValueError, error_msg

    def _generate_all_conditions(self):
        """
        generate all possible conditions for one iteration given the lists of 
        parameters
        
        Returns
        -------
        all_conditions : list of tuples
             all unique combinations of spatial frequency, temporal frequency,
             direction, contrast, and size. Output depends on initialization
             parameters.
            
        """
        all_conditions = [(sf, tf, dire, con, size) for sf in self.sf_list
                                                    for tf in self.tf_list
                                                    for dire in self.dire_list
                                                    for con in self.con_list
                                                    for size in self.size_list]
        random.shuffle(all_conditions)

        return all_conditions

    def _generate_phase_list(self, tf):
        """
        get a list of phases that will be displayed for each frame in the block 
        duration, also make the first frame of each cycle
        
        Parameters
        ----------
        tf : float
            temporal frequency in Hz

        Returns
        -------
        phases : 
            list of phases in one block
        frame_per_cycle : 
            number of frames for each circle
        """
        block_frame_num = int(self.block_dur * self.monitor.refresh_rate)

        frame_per_cycle = int(self.monitor.refresh_rate / tf)

        phases_per_cycle = list(np.arange(0,np.pi*2,np.pi*2/frame_per_cycle))

        phases = []

        while len(phases) < block_frame_num:
            phases += phases_per_cycle

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
        returns a list of information of all frames as a list of tuples

        Information contained in each frame:
             first element: 
                  value equal to 1 during stimulus and 0 otherwise
             second element: 
                  on first frame in a cycle value takes on 1, and otherwise is
                  equal to 0.
             third element: 
                  spatial frequency
             forth element: 
                  temporal frequency
             fifth element: 
                  direction, [0, 2*pi)
             sixth element: 
                  contrast
             seventh element: 
                  size (radius of the circle)
             eighth element: 
                  phase, [0, 2*pi)
             ninth element: 
                  indicator color [-1, 1]. Value is equal to 1 on the first
                  frame of each cycle, -1 during gaps and otherwise 0.
        
             during gap frames the second through the eighth elements should 
             be 'None'.
        """
        frames = []
        off_params = [0, None,None,None,None,None,None,None,-1.]

        for i in range(self.iteration):
            if i == 0: # very first block
                frames += [off_params for ind in range(self.pregap_frame_num)]
            else: # first block for the later iteration
                frames += [off_params for ind in range(int(self.midgap_dur * self.monitor.refresh_rate))]

            all_conditions = self._generate_all_conditions()

            for j, condition in enumerate(all_conditions):
                if j != 0: # later conditions
                    frames += [off_params for ind in range(int(self.midgap_dur * self.monitor.refresh_rate))]

                sf, tf, dire, con, size = condition

                # get phase list for each condition
                phases, frame_per_cycle = self._generate_phase_list(tf)
                if (dire % (np.pi * 2)) >= np.pi: 
                     phases = [-phase for phase in phases]

                for k, phase in enumerate(phases): # each frame in the block

                    # mark first frame of each cycle
                    if k % frame_per_cycle == 0:
                        first_in_cycle = 1
                    else:
                        first_in_cycle = 0

                    frames.append([1,first_in_cycle,sf,tf,dire,con,size,phase,float(first_in_cycle)])

        # add post gap frame
        frames += [off_params for ind in range(self.postgap_frame_num)]

        #add non-synchronized indicator
        if self.indicator.is_sync == False:
            for l in range(len(frames)):
                if np.floor(l // self.indicator.frame_num) % 2 == 0:
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
        if self.coordinate=='degree':
             coord_x=self.monitor.deg_coord_x
             coord_y=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_x=self.monitor.lin_coord_x
             coord_y=self.monitor.lin_coord_y
        
        for size in self.size_list:
            curr_mask = get_circle_mask(coord_x, coord_y, self.center, size)
            masks.update({size:curr_mask})
            
        return masks

    def generate_movie(self):
        """
        Generate movie frame by frame
        """
        
        self.frames = self.generate_frames()
        mask_dict = self._generate_circle_mask_dict()

        if self.coordinate=='degree':
             coord_x=self.monitor.deg_coord_x
             coord_y=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_x=self.monitor.lin_coord_x
             coord_y=self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        indicator_width_min = (self.indicator.center_width_pixel - 
                               (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel + 
                               (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel - 
                                (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel + 
                                (self.indicator.height_pixel / 2))

        mov = np.ones((len(self.frames),
                       coord_x.shape[0],
                       coord_x.shape[1]),dtype=np.float16) * self.background
        background_frame = np.ones(coord_x.shape,dtype=np.float16)*self.background

        for i, curr_frame in enumerate(self.frames):

            if curr_frame[0] == 1: # not a gap

                curr_ori = self._get_ori(curr_frame[4])

                curr_grating = get_grating(coord_x,
                                           coord_y,
                                           ori = curr_ori,
                                           spatial_freq = curr_frame[2],
                                           center = self.center,
                                           phase = curr_frame[7],
                                           contrast = curr_frame[5])
                curr_grating = curr_grating*2. - 1.

                curr_circle_mask = mask_dict[curr_frame[6]]

                mov[i] = ((curr_grating * curr_circle_mask) + 
                             (background_frame * (curr_circle_mask * -1. + 1.)))

            #add sync square for photodiode
            mov[i, indicator_height_min:indicator_height_max, 
                indicator_width_min:indicator_width_max] = curr_frame[-1]

            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+
                       str(int(100 * (i+1) / len(self.frames)))+'%']

        #generate log dictionary
        mondict=dict(self.monitor.__dict__)
        indicator_dict=dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        self_dict=dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')
        log={'stimulation':self_dict,
             'monitor':mondict,
             'indicator':indicator_dict}

        return mov, log
    
class KSstim(Stim):
    """
    generate Kalatsky & Stryker stimulus, integrates flashing indicator for 
    photodiode
    """
    def __init__(self,
                 monitor,
                 indicator,
                 background=0., 
                 coordinate='degree',
                 square_size=25.,
                 square_center=(0,0), 
                 flicker_frame=10,
                 sweep_width=20., 
                 step_width=0.15, 
                 direction='B2U', 
                 sweep_frame=1,
                 iteration=1, 
                 pregap_dur=2.,
                 postgap_dur=3.):
       
        super(KSstim,self).__init__(monitor=monitor,
                                    indicator=indicator,
                                    coordinate=coordinate,
                                    background=background,
                                    pregap_dur=pregap_dur,
                                    postgap_dur=postgap_dur)
        """
        Initialize Kalatsky & Stryker stimulus object
        
        Attributes
        ----------
        monitor : monitor object
            object storing experimental monitor setup information
        indicator : indicator object
            object storing photodiode indicator information
        background : float, optional
            background color of stimulus, takes values in [-1,1]. defaults to 
            `0.`
        coordinate : str, optional
            coordinate representation, either 'degree' or 'linear', defaults
            to 'degree'
        square_size : float, optional
            size of flickering square, defaults to `25.`
        square_center: tuple, optional
            coordinate of center point, defaults to `(0,0)`
        flicker_frame : int, optional
            defaults to `10`
        sweep_width : float, optional
            width of sweeps measured in units cm or degs if coordinate value
            is 'linear' or 'degree' respectively. defaults to `20`
        step_width : float, optional
            width of steps measured in units cm or degs if coordinate value
            is 'linear' or 'degree' respectively. defaults to `0.15`
        direction : {'B2U','U2B','L2R','R2L'}, optional
            the direction of sweep movement, defaults to 'B2U'. 'B2U' means 
            stim is presented from the bottom to the top of the screen, whereas
            'U2B' is from the top to the bottom. 'L2R' is left to right and 'R2L'
            is right to left
        sweep_frame : int, optional
            defaults to 1
        iteration : int, optional
            number of times that the stimulus will be repeated, defaults to `1`
        pregap_dur : float, optional
            number of seconds before stimulus is presented, defaults to `2` 
        postgap_dur : float, optional
            number of seconds after stimulus is presented, defaults to `2` 
        """
                     
        self.stim_name = 'KSstim'
        self.square_size = square_size
        self.square_center = square_center
        self.flicker_frame = flicker_frame
        self.flicker_freq = self.monitor.refresh_rate / self.flicker_frame
        self.sweep_width = sweep_width
        self.step_width = step_width
        self.direction = direction
        self.sweep_frame = sweep_frame
        self.iteration = iteration
        self.frame_config = ('is_display', 'squarePolarity', 
                            'sweep_index', 'indicator_color')
        self.sweep_config = ('orientation', 'sweepStartCoordinate', 
                            'sweepEndCoordinate')
        
        self.sweep_speed = (self.monitor.refresh_rate * 
                           self.step_width / self.sweep_frame) 
        self.flicker_hz = self.monitor.refresh_rate / self.flicker_frame

        self.clear()
        

    def generate_squares(self):
        """
        generate checker board squares
        """
        
        if self.coordinate == 'degree':
            map_x = self.monitor.deg_coord_x
            map_y = self.monitor.deg_coord_y
            
        elif self.coordinate == 'linear':
            map_x = self.monitor.lin_coord_x
            map_y = self.monitor.lin_coord_y
            
        else:
            raise LookupError, '`coordinate` not in {"degree","linear"}'
        
        min_x = map_x.min()
        max_x = map_x.max()
        
        min_y = map_y.min()
        max_y = map_y.max()
        
        neg_x = np.ceil( abs( ( ( min_x - self.square_center[0] ) / 
                              ( 2 * self.square_size ) ) ) ) + 1
        pos_x = np.ceil( abs( ( ( max_x - self.square_center[0] ) / 
                              ( 2 * self.square_size ) ) ) ) + 1
        
        neg_y = np.ceil( abs( ( ( min_y - self.square_center[0] ) / 
                              ( 2 * self.square_size ) ) ) ) + 1
        pos_y = np.ceil( abs( ( ( max_y - self.square_center[0] ) / 
                              ( 2 * self.square_size ) ) ) ) + 1
        
        squareV = np.ones((np.size(map_x, 0), 
                           np.size(map_x, 1)), 
                           dtype = np.float16)
        squareV = -1 * squareV
        
        stepV = np.arange(self.square_center[0] - (2*neg_x + 0.5)*self.square_size, 
                          self.square_center[0] + (2*pos_x - 0.5)*self.square_size, 
                          self.square_size*2)
        
        for i in range(len(stepV)):
            squareV[np.where(np.logical_and(map_x >= stepV[i],
                                            map_x < (stepV[i] + 
                                                     self.square_size)))] = 1.0
        
        squareH = np.ones((np.size(map_y, 0), 
                           np.size(map_y, 1)), dtype = np.float16)
        squareH = -1 * squareH
        
        stepH = np.arange(self.square_center[1] - (2*neg_y + 0.5)*self.square_size, 
                          self.square_center[1] + (2*pos_y - 0.5)*self.square_size, 
                          self.square_size*2)
        
        for j in range(len(stepH)):
            squareH[ np.where(np.logical_and(map_y >= stepH[j], 
                                             map_y < (stepH[j] + 
                                                      self.square_size)))] = 1
        
        squares = np.multiply(squareV, squareH)
        
        return squares

    def plot_squares(self):
        """
        plot checkerboard squares
        """
        plt.figure()
        plt.imshow(self.squares)

    def generate_sweeps(self):
        """
        generate full screen sweep sequence
        """
        sweep_width = self.sweep_width
        step_width =  self.step_width
        direction = self.direction
        
        if self.coordinate == 'degree':
            map_x = self.monitor.deg_coord_x
            map_y = self.monitor.deg_coord_y
            
        elif self.coordinate == 'linear':
            map_x = self.monitor.lin_coord_x
            map_y = self.monitor.lin_coord_y
        else:
            raise LookupError, '`coordinate` not in {"degree", "linear"}'
        
        min_x = map_x.min()
        max_x = map_x.max()
        
        min_y = map_y.min()
        max_y = map_y.max()
        
        if direction == "B2U":
            step_y = np.arange(min_y - sweep_width, 
                               max_y + step_width, step_width)
        elif direction == "U2B":
            step_y = np.arange(min_y - sweep_width, 
                               max_y + step_width, step_width)[::-1]
        elif direction == "L2R":
            step_x = np.arange(min_x - sweep_width, 
                               max_x + step_width, step_width)
        elif direction == "R2L":
            step_x = np.arange(min_x - sweep_width, 
                               max_x + step_width, step_width)[::-1]
        else:
            raise LookupError, '`direction` not in {"B2U","U2B","L2R","R2L"}'
        
        sweep_table = []
        
        if 'step_x' in locals():
            sweeps = np.zeros((len(step_x), 
                               np.size(map_x, 0), 
                               np.size(map_x, 1)), dtype = np.float16)
            for i in range(len(step_x)):
                temp = sweeps[i,:,:]
                temp[np.where(np.logical_and(map_x >= step_x[i], 
                                             map_x < (step_x[i] + 
                                                      sweep_width)))] = 1.0
                sweep_table.append(('V', step_x[i], step_x[i] + sweep_width))
                del temp
                
        if 'step_y' in locals():
            sweeps = np.zeros((len(step_y), 
                               np.size(map_y, 0), 
                               np.size(map_y, 1)), dtype = np.float16)
            for j in range(len(step_y)):
                temp=sweeps[j,:,:]
                temp[np.where(np.logical_and(map_y >= step_y[j], 
                                             map_y < (step_y[j] + 
                                                      sweep_width)))] = 1.0
                sweep_table.append(('H', step_y[j], step_y[j] + sweep_width))
                del temp
                
        return sweeps.astype(np.bool), sweep_table

    def generate_frames(self):
        """
        function to generate all the frames needed for KS stimulation
        returnins a list of information of all frames as a list of tuples
        
        Information contained in each frame:
             first element: 
                  during stimulus value is equal to 1 and 0 otherwise
             second element: 
                  square polarity, 1->not reversed; -1->reversed
             third element: 
                  sweeps, index in sweep table
             forth element: 
                  color of indicator
                       synchronized: gap->-1, sweep on -> 1
                       non-synchronized: alternating between -1 and 1 at defined frequency
        
          for gap frames the second and third elements should be 'None'
        """
        
        sweeps, _ = self.generate_sweeps()
        sweep_frame = self.sweep_frame
        flicker_frame = self.flicker_frame
        iteration = self.iteration
        
        sweep_num = np.size(sweeps,0) # Number of sweeps vertical or horizontal
        displayframe_num = sweep_frame*sweep_num # total frame number for 1 iter
                                                                              
        
        #frames for one iteration
        iter_frames=[] 
        
        #add frames for gaps
        for i in range(self.pregap_frame_num):
            iter_frames.append([0,None,None,-1])
        
        
        #add frames for display
        is_reverse=[]
        
        for i in range(displayframe_num):
            
            if (np.floor(i // flicker_frame)) % 2 == 0:
                is_reverse = -1
            else:
                is_reverse = 1
                
            sweep_index=int(np.floor(i // sweep_frame))
            
            #add sychornized indicator
            if self.indicator.is_sync == True:
                indicator_color = 1
            else:
                indicator_color = -1
                
            iter_frames.append([1,is_reverse,sweep_index,indicator_color])
            
        # add gap frames at the end
        for i in range(self.postgap_frame_num):
            iter_frames.append([0,None,None,-1])
        
        full_frames = []
        
        #add frames for multiple iteration
        for i in range(int(iteration)):
            full_frames += iter_frames
        
        #add non-synchronized indicator
        if self.indicator.is_sync == False:
            indicator_frame = self.indicator.frame_num
            
            for j in range(np.size(full_frames,0)):
                if np.floor(j // indicator_frame) % 2 == 0:
                    full_frames[j][3] = 1
                else:
                    full_frames[j][3] = -1
            
        full_frames = [tuple(x) for x in full_frames]
        
        return tuple(full_frames)

    def generate_movie(self):
        """
        Function to Generate Kalatsky & Stryker visual stimulus frame by frame
        """
        
        self.squares = self.generate_squares()
        
        sweeps, self.sweep_table = self.generate_sweeps()

        self.frames=self.generate_frames()
        
        full_seq = np.zeros((len(self.frames),
                                 self.monitor.deg_coord_x.shape[0],
                                 self.monitor.deg_coord_x.shape[1]),
                                                         dtype=np.float16)
        
        indicator_width_min = (self.indicator.center_width_pixel - 
                               (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel + 
                               (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel - 
                                (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel + 
                                (self.indicator.height_pixel / 2))
        
        background = np.ones((np.size(self.monitor.deg_coord_x, 0), 
                              np.size(self.monitor.deg_coord_x,1)), 
                                 dtype = np.float16) * self.background 
        
        for i in range(len(self.frames)):
            curr_frame = self.frames[i]
            
            if curr_frame[0] == 0:
                curr_NM_seq = background
                
            else:
                currSquare = self.squares * curr_frame[1]
                curr_sweep = sweeps[curr_frame[2]]
                curr_NM_seq = ((curr_sweep * currSquare) + 
                                    ((-1 * (curr_sweep - 1)) * background))

            curr_NM_seq[indicator_height_min:indicator_height_max, 
                        indicator_width_min:indicator_width_max] = curr_frame[3]
            
            full_seq[i] = curr_NM_seq
            
            if i in range(0, len(self.frames),len(self.frames)/10):
                print ['Generating numpy sequence: '+str(int(100 * (i+1) 
                                                   / len(self.frames)))+'%']
        
        
        mondict=dict(self.monitor.__dict__)
        indicator_dict=dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')        
        KSdict=dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        full_dict={'stimulation':KSdict, 
                        'monitor':mondict,
                        'indicator':indicator_dict} 
                        
        return full_seq, full_dict

    def clear(self):
        self.sweep_table = None
        self.frames = None
        self.square = None

    def set_direction(self,direction):
        
        if direction in ['B2U','U2B','L2R','R2L']: 
            self.direction = direction
            self.clear()
        else:
            raise LookupError, '`direction` not in {"B2U","U2B","L2R","R2L"}'

    def set_sweep_sigma(self,sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()

    def set_sweep_width(self,sweep_width):
        self.sweep_width = sweep_width
        self.clear()
   
class KSstimAllDir(object):
    """
    generate Kalatsky & Stryker stimulation in all four direction contiuously
    """
    def __init__(self,
                 monitor,
                 indicator,
                 coordinate='degree', 
                 background=0., 
                 square_size=25, 
                 square_center=(0,0), 
                 flicker_frame=6,
                 sweep_width=20., 
                 step_width=0.15,
                 sweep_frame=1,
                 iteration=1,
                 pregap_dur=2.,
                 postgap_dur=3.):
        
        """
        Initialize stimulus
        
        Parameters
        ----------
        monitor : monitor object
            contains display monitor information
        indicator : indicator object
            contains indicator information
        coordinate : str from {'degree','linear'}, optional
            specifies coordinates, defaults to 'degree'
        background : float, optional
            color of background. Takes values in [-1,1] where -1 is black and 1
            is white
        square_size : int, optional
            size of flickering square, defaults to 25.
        square_center : tuple, optional
            coordinate of center point of the square, defaults to (0,0)
        flicker_frame : int, optional
            ... defaults to 6
        sweep_width : float, optional
            width of sweeps. defaults to 20.
        step_width : float, optional
            width of steps. defaults to 0.15.
        sweep_frame : int, optional
            ... defaults to 1
        iteration : int, optional
            number of times stimulus will be presented, defaults to 1
        pregap_dur : float, optional
            number of seconds before stimulus is presented
        postgap_dur : float, optional
            number of seconds after stimulus is presented
        """

        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate
        self.square_size = square_size
        self.square_center = square_center
        self.flicker_frame = flicker_frame
        self.sweep_width = sweep_width
        self.step_width = step_width
        self.sweep_frame = sweep_frame
        self.iteration = iteration
        self.pregap_dur = pregap_dur
        self.postgap_dur = postgap_dur


    def generate_movie(self):
        """
        Generate stimulus movie frame by frame
        """
        KS_stim=KSstim(self.monitor,
                       self.indicator,
                       background=self.background,
                       coordinate=self.coordinate,
                       direction='B2U',
                       square_size=self.square_size,
                       square_center=self.square_center,
                       flicker_frame=self.flicker_frame,
                       sweep_width=self.sweep_width,
                       step_width=self.step_width,
                       sweep_frame=self.sweep_frame,
                       iteration=self.iteration,
                       pregap_dur=self.pregap_dur,
                       postgap_dur=self.postgap_dur)

        mov_B2U, dict_B2U = KS_stim.generate_movie()
        KS_stim.set_direction('U2B')
        mov_U2B, dict_U2B = KS_stim.generate_movie()
        KS_stim.set_direction('L2R')
        mov_L2R, dict_L2R = KS_stim.generate_movie()
        KS_stim.set_direction('R2L')
        mov_R2L, dict_R2L = KS_stim.generate_movie()

        mov = np.vstack((mov_B2U,mov_U2B,mov_L2R,mov_R2L))
        log = {'monitor':dict_B2U['monitor'],
               'indicator':dict_B2U['indicator']}
        stimulation = dict(dict_B2U['stimulation'])
        stimulation['stim_name'] = 'KSstimAllDir'
        stimulation['direction'] = ['B2U','U2B','L2R','R2L']

        sweep_table = []
        frames = []

        sweep_table_B2U = dict_B2U['stimulation']['sweep_table']
        frames_B2U = dict_B2U['stimulation']['frames']
        sweep_length_B2U = len(sweep_table_B2U)
        sweep_table_B2U = [ ['B2U', x[1], x[2]] for x in sweep_table_B2U] 
        frames_B2U = [[x[0],x[1],x[2],x[3],'B2U'] for x in frames_B2U]
        sweep_table += sweep_table_B2U
        frames += frames_B2U

        sweep_table_U2B = dict_U2B['stimulation']['sweep_table']
        frames_U2B = dict_U2B['stimulation']['frames'] 
        sweep_length_U2B = len(sweep_table_U2B)
        sweep_table_U2B = [ ['U2B', x[1], x[2]] for x in sweep_table_U2B]
        frames_U2B = [[x[0],x[1],x[2],x[3],'U2B'] for x in frames_U2B]
        for frame in frames_U2B:
            if frame[2] is not None: 
                 frame[2] += sweep_length_B2U
        sweep_table += sweep_table_U2B 
        frames += frames_U2B

        sweep_table_L2R = dict_L2R['stimulation']['sweep_table']
        frames_L2R = dict_L2R['stimulation']['frames']
        sweep_length_L2R = len(sweep_table_L2R)
        sweep_table_L2R = [ ['L2R', x[1], x[2]] for x in sweep_table_L2R]
        frames_L2R = [[x[0],x[1],x[2],x[3],'L2R'] for x in frames_L2R]
        for frame in frames_L2R:
            if frame[2] is not None: 
                 frame[2] += sweep_length_B2U+sweep_length_U2B
        sweep_table += sweep_table_L2R
        frames += frames_L2R

        sweep_table_R2L = dict_R2L['stimulation']['sweep_table']
        frames_R2L = dict_R2L['stimulation']['frames']
        sweep_table_R2L = [ ['R2L', x[1], x[2]] for x in sweep_table_R2L] 
        frames_R2L = [[x[0],x[1],x[2],x[3],'R2L'] for x in frames_R2L]
        for frame in frames_R2L:
            if frame[2] is not None: 
                 frame[2] += sweep_length_B2U+sweep_length_U2B+sweep_length_L2R
        sweep_table += sweep_table_R2L 
        frames += frames_R2L

        stimulation['frames'] = [tuple(x) for x in frames]
        stimulation['sweep_table'] = [tuple(x) for x in sweep_table]

        log['stimulation'] = stimulation
        log['stimulation']['frame_config'] = ('is_display', 'squarePolarity',
                                              'sweep_index', 'indicator_color')
        log['stimulation']['sweep_config'] = ('orientation', 
                                              'sweepStartCoordinate', 
                                              'sweepEndCoordinate')

        return mov, log

class DisplaySequence(object):
    """
    Display the numpy sequence from memory
    """        
    
    def __init__(self,
                 log_dir,
                 backupdir=None,
                 display_iter=1,
                 display_order=1,  
                 mouse_id='Test',
                 user_id='Name',
                 psychopy_mon='testMonitor',
                 is_interpolate=False,
                 is_triggered=False,
                 trigger_NI_dev='Dev1',
                 trigger_NI_port=1,
                 trigger_NI_line=0,
                 is_sync_pulse_pulse=True,
                 sync_pulse_NI_dev='Dev1',
                 sync_pulse_NI_port=1,
                 sync_pulse_NI_line=1,
                 display_trigger_event="negative_edge", 
                 display_screen=0,
                 initial_background_color=0,
                 file_num_NI_dev='Dev1',
                 file_num_NI_port='0',
                 file_num_NI_lines='0:7'):
        """
        initialize `DisplaySequence` object
        
        Parameters
        ----------
        log_dir : str
            system directory path to where log display will be saved
        backupdir : str, optional
            copy of directory path to save backup, defaults to `None`
        display_iter : int, optional
            defaults to 1
        display_order : int, optional
            determines whether the stimulus is presented forward or backwards.
            If 1, stimulus is presented forward, whereas if -1, stimulus is 
            presented backwards. Defaults to 1.
        mouse_id : str, optional
            label for mouse, defaults to 'Test'
        user_id : str, optional
            label for person performing experiment, defaults to 'Name'
        psychopy_mon : str, optional
            label for monitor used for displaying the stimulus, defaults to 
            'testMonitor'
        is_interpolate : bool, optional
            defaults to False
        is_triggered : bool, optional
            if True, stimulus will not display until triggered. if False, 
            stimulus will display automatically. defaults to False
        trigger_NI_dev : str, optional
            defaults to 'Dev1'
        trigger_NI_port : int, optional
            defaults to 1
        trigger_NI_line : int, optional
            defaults to 0
        is_sync_pulse_pulse : bool, optional
            defaults to True
        sync_pulse_NI_dev : str, optional
            defaults to 'Dev1'
        sync_pulse_NI_port : int, optional 
            defaults to 1
        sync_pulse_NI_line : int, optional
            defaults to 1
        display_trigger_event : 
            should be one of "negative_edge", "positive_edge", "high_level", 
            or "low_level". defaults to "negative_edge"
        display_screen : 
            determines which monitor to display stimulus on. defaults to 0
        initial_background_color : 
            defaults to 0
        file_num_NI_dev : 
            defaults to 'Dev1',
        file_num_NI_port : 
            defaults to '0'
        file_num_NI_lines : 
            defaults to '0:7'
        """
        self.sequence = None
        self.seq_log = {}
        self.psychopy_mon = psychopy_mon
        self.is_interpolate = is_interpolate
        self.is_triggered = is_triggered
        self.trigger_NI_dev = trigger_NI_dev
        self.trigger_NI_port = trigger_NI_port
        self.trigger_NI_line = trigger_NI_line
        self.display_trigger_event = display_trigger_event
        self.is_sync_pulse_pulse = is_sync_pulse_pulse
        self.sync_pulse_NI_dev = sync_pulse_NI_dev
        self.sync_pulse_NI_port = sync_pulse_NI_port
        self.sync_pulse_NI_line = sync_pulse_NI_line
        self.display_screen = display_screen
        self.initial_background_color = initial_background_color
        self.keep_display = None
        self.file_num_NI_dev = file_num_NI_dev
        self.file_num_NI_port = file_num_NI_port
        self.file_num_NI_lines = file_num_NI_lines

        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."
            
        self.display_order = display_order
        self.log_dir = log_dir
        self.backupdir = backupdir
        self.mouse_id = mouse_id
        self.user_id = user_id
        self.seq_log = None
                
        self.clear()

    def set_any_array(self, any_array, log_dict = None):
        """
        to display any numpy 3-d array.
        """
        if len(any_array.shape) != 3:
            raise LookupError, "Input numpy array should have dimension of 3!"
        
        vmax = np.amax(any_array).astype(np.float32)
        vmin = np.amin(any_array).astype(np.float32)
        v_range = (vmax-vmin)
        any_array_nor = ((any_array-vmin)/v_range).astype(np.float16)
        self.sequence = 2*(any_array_nor-0.5)

        if log_dict != None:
            if type(log_dict) is dict:
                self.seq_log = log_dict
            else:
                raise ValueError, '`log_dict` should be a dictionary!'
        else:
            self.seq_log = {}
        self.clear()
    

    def set_stim(self, stim):
        """
        generate sequence of stimulus to be displayed.
        
        Calls the `generate_movie` method of the respective stim object and 
        populates the attributes `self.sequence` and `self.seq_log`
        
        Parameters
        ----------
        stim : Stim object
            the type of stimulus to be presented in the experiment
        """
        self.sequence, self.seq_log = stim.generate_movie()
        self.clear()


    def trigger_display(self):
        """
        Display stimulus
        
        Prepares all of the necessary parameters to display stimulus and store
        the data collected. 
        """
        # --------------- early preparation for display--------------------
        # test monitor resolution
        try: 
             resolution = self.seq_log['monitor']['resolution'][::-1]
        except KeyError: 
             resolution = (800,600)

        # test monitor refresh rate
        try:
            refresh_rate = self.seq_log['monitor']['refresh_rate']
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz.\n"
            refresh_rate = 60.

        #prepare display frames log
        if self.sequence is None:
            raise LookupError, "Please set the sequence to be displayed!!\n"
        try:
            seq_frames = self.seq_log['stimulation']['frames']
            if self.display_order == -1: 
                 seq_frames = seq_frames[::-1]
            # generate display Frames
            self.display_frames=[]
            for i in range(self.display_iter):
                self.display_frames += seq_frames
        except Exception as e:
            print e
            print "No frame information in seq_log dictionary."
            print "Setting display_frames to 'None'.\n"
            self.display_frames = None

        # calculate expected display time
        display_time = (float(self.sequence.shape[0]) * 
                        self.display_iter / refresh_rate)
        print '\n Expected display time: ', display_time, ' seconds\n'

        # generate file name
        self._get_file_name()
        print 'File name:', self.file_name + '\n'
        

        # -----------------setup psychopy window and stimulus--------------
        # start psychopy window
        window = visual.Window(size=resolution,
                               monitor=self.psychopy_mon,
                               fullscr=True,
                               screen=self.display_screen,
                               color=self.initial_background_color)
        stim = visual.ImageStim(window, size=(2,2), 
                                interpolate=self.is_interpolate)
       
        # initialize keep_display
        self.keep_display = True

        # handle display trigger
        if self.is_triggered:
            display_wait = self._wait_for_trigger(event=self.display_trigger_event)
            if not display_wait:
                window.close()
                self.clear()
                return None
            else:
                time.sleep(5.) # wait remote object to start

        # display sequence
        self._display(window, stim) 

        self.save_log()

        #analyze frames
        try: 
             self.frame_duration, self.frame_stats = \
             analyze_frames(ts=self.time_stamp, 
                            refresh_rate = self.seq_log['monitor']['refresh_rate'])
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz."
            self.frame_duration, self.frame_stats = \
                 analyze_frames(ts = self.time_stamp, refresh_rate = 60.)

        #clear display data
        self.clear()

    def _wait_for_trigger(self, event):
        """
        time place holder for waiting for trigger

        Parameters
        ----------
        event : str from {'low_level','high_level','negative_edge','positive_edge'}

        Returns
        -------
        Bool :
             returns True if trigger is detected and False if manual stop 
             signal is detected
        """

        #check NI signal
        trigger_task = iodaq.DigitalInput(self.trigger_NI_dev, 
                                         self.trigger_NI_port, 
                                         self.trigger_NI_line)
        trigger_task.StartTask()

        print "Waiting for trigger: " + event + ' on ' + trigger_task.devstr

        if event == 'low_level':
            last_TTL = trigger_task.read()
            while last_TTL != 0 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display: 
                     trigger_task.StopTask()
                     print 'Trigger detected. Start displaying...\n\n'
                     return True
                else:
                     trigger_task.StopTask()
                     print 'Manual stop signal detected. Stopping the program.'
                     return False
        elif event == 'high_level':
            last_TTL = trigger_task.read()[0]
            while last_TTL != 1 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display: 
                     trigger_task.StopTask()
                     print 'Trigger detected. Start displaying...\n\n'
                     return True
                else: 
                     trigger_task.StopTask()
                     print 'Manual stop signal detected. Stopping the program.'
                     return False
        elif event == 'negative_edge':
            last_TTL = trigger_task.read()[0]
            while self.keep_display:
                current_TTL = trigger_task.read()[0]
                if (last_TTL == 1) and (current_TTL == 0):
                     break
                else:
                     last_TTL = int(current_TTL)
                     self._update_display_status()
            else: 
                 trigger_task.StopTask()
                 print 'Manual stop signal detected. Stopping the program.'
                 return False
            trigger_task.StopTask()
            print 'Trigger detected. Start displaying...\n\n'
            return True
        elif event == 'positive_edge':
            last_TTL = trigger_task.read()[0]
            while self.keep_display:
                current_TTL = trigger_task.read()[0]
                if (last_TTL == 0) and (current_TTL == 1):
                     break
                else:
                     last_TTL = int(current_TTL)
                     self._update_display_status()
            else: 
                 trigger_task.StopTask(); 
                 print 'Manual stop signal detected. Stopping the program.'
                 return False
            trigger_task.StopTask()
            print 'Trigger detected. Start displaying...\n\n'
            return True
        else:
             raise NameError, "`trigger` not in " \
             "{'negative_edge','positive_edge', 'high_level','low_level'}!"


    def _get_file_name(self):
        """
        generate the file name of log file
        """

        try:
            self.file_name = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + \
                            self.seq_log['stimulation']['stim_name'] + \
                            '-M' + \
                            self.mouse_id + \
                            '-' + \
                            self.user_id
        except KeyError:
            self.file_name = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + 'customStim' + '-M' + self.mouse_id + '-' + \
                            self.user_id
        
        file_number = self._get_file_number()
        
        if self.is_triggered: 
             self.file_name += '-' + str(file_number)+'-Triggered'
        else: 
             self.file_name += '-' + str(file_number) + '-notTriggered'


    def _get_file_number(self):
        """
        get synced file number for log file name
        """
        
        try:
            file_num_task = iodaq.DigitalInput(self.file_num_NI_dev,
                                             self.file_num_NI_port,
                                             self.file_num_NI_lines)
            file_num_task.StartTask()
            array = file_num_task.read()
            num_str = (''.join([str(line) for line in array]))[::-1]
            file_number = int(num_str, 2)
            # print array, file_number
        except Exception as e:
            print e
            file_number = None

        return file_number
        

    def _display(self, window, stim):
        
        # display frames
        time_stamp=[]
        start_time = time.clock()
        singleRunFrames = self.sequence.shape[0]
        
        if self.is_sync_pulse_pulse:
            syncPulseTask = iodaq.DigitalOutput(self.sync_pulse_NI_dev, 
                                                self.sync_pulse_NI_port, 
                                                self.sync_pulse_NI_line)
            syncPulseTask.StartTask()
            _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

        i = 0

        while self.keep_display and i < (singleRunFrames * self.display_iter):

            if self.display_order == 1:
                 frame_num = i % singleRunFrames

            if self.display_order == -1:
                 frame_num = singleRunFrames - (i % singleRunFrames) -1

            stim.setImage(self.sequence[frame_num][::-1,:])
            stim.draw()
            time_stamp.append(time.clock()-start_time)

            #set syncPuls signal
            if self.is_sync_pulse_pulse: 
                 _ = syncPulseTask.write(np.array([1]).astype(np.uint8))

            #show visual stim
            window.flip()
            #set syncPuls signal
            if self.is_sync_pulse_pulse: 
                 _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

            self._update_display_status()
            i=i+1
            
        stop_time = time.clock()
        window.close()
        
        if self.is_sync_pulse_pulse:
             syncPulseTask.StopTask()
        
        self.time_stamp = np.array(time_stamp)
        self.display_length = stop_time-start_time

        if self.display_frames is not None:
            self.display_frames = self.display_frames[:i]

        if self.keep_display == True: 
             print '\nDisplay successfully completed.'


    def flag_to_close(self):
        self.keep_display = False


    def _update_display_status(self):

        if self.keep_display is None: 
             raise LookupError, 'self.keep_display should start as True'

        #check keyboard input 'q' or 'escape'
        keyList = event.getKeys(['q','escape'])
        if len(keyList) > 0:
            self.keep_display = False
            print "Keyboard stop signal detected. Stop displaying. \n"


    def set_display_order(self, display_order):
        
        self.display_order = display_order
        self.clear()
    

    def set_display_iteration(self, display_iter):
        
        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."
        self.clear()
        

    def save_log(self):
        
        if self.display_length is None:
            self.clear()
            raise LookupError, "Please display sequence first!"

        if self.file_name is None:
            self._get_file_name()
            
        if self.keep_display == True:
            self.file_name += '-complete'
        elif self.keep_display == False:
            self.file_name += '-incomplete'

        #set up log object
        directory = self.log_dir + '\sequence_display_log'
        if not(os.path.isdir(directory)):
             os.makedirs(directory)
        
        logFile = dict(self.seq_log)
        displayLog = dict(self.__dict__)
        displayLog.pop('seq_log')
        displayLog.pop('sequence')
        logFile.update({'presentation':displayLog})

        file_name =  self.file_name + ".pkl"
        
        #generate full log dictionary
        path = os.path.join(directory, file_name)
        ft.saveFile(path,logFile)
        print ".pkl file generated successfully."
        
        backupFileFolder = self._get_backup_folder()
        if backupFileFolder is not None:
            if not (os.path.isdir(backupFileFolder)): 
                 os.makedirs(backupFileFolder)
            backupFilePath = os.path.join(backupFileFolder,file_name)
            ft.saveFile(backupFilePath,logFile)
            print ".pkl backup file generate successfully"
        else:
            print "did not find backup path, no backup has been saved."
            
    def _get_backup_folder(self):
        
        if self.file_name is None:
            raise LookupError, 'self.file_name not found.'
        else:
        
            if self.backupdir is not None:

                curr_date = self.file_name[0:6]
                stim_name = self.seq_log['stimulation']['stim_name']
                if 'KSstim' in stim_name:
                    backupFileFolder = \
                         os.path.join(self.backupdir,
                                      curr_date+'-M'+self.mouse_id+'-Retinotopy')
                else:
                    backupFileFolder = \
                         os.path.join(self.backupdir,
                                      curr_date+'-M'+self.mouse_id+'-'+stim_name)
                return backupFileFolder
            else:
                return None


    def clear(self):
        """ clear display information. """
        self.display_length = None
        self.time_stamp = None
        self.frame_duration = None
        self.display_frames = None
        self.frame_stats = None
        self.file_name = None
        self.keep_display = None


if __name__ == "__main__":

    #==============================================================================================================================
     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=5)
     indicator=Indicator(mon)
     KS_stim=KSstim(mon,indicator)
     display_iter = 2
     # print (len(KSstim.generate_frames())*display_iter)/float(mon.refresh_rate)
     ds=DisplaySequence(log_dir=r'C:\data',backupdir=r'C:\data',is_triggered=True,display_iter=2,display_screen=1)
     ds.set_stim(KS_stim)
     ds.trigger_display()
     plt.show()
    #==============================================================================================================================

   
    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=10)
#     indicator=Indicator(mon)
#     flashing_circle=FlashingCircle(mon,indicator)
#     display_iter = 2
#     print (len(flashing_circle.generate_frames())*display_iter)/float(mon.refresh_rate)
#     ds=DisplaySequence(log_dir=r'C:\data',backupdir=r'C:\data',is_triggered=True,display_iter=2,display_screen=1)
#     ds.set_stim(flashing_circle)
#     ds.trigger_display()
#     plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#     mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#     indicator=Indicator(mon)
#     sparse_noise=SparseNoise(mon,indicator, subregion=(-20.,20.,40.,60.), grid_space=(10, 10))
#     grid_points = sparse_noise._generate_grid_points_sequence()
#     gridLocations = np.array([l[0] for l in grid_points])
#     plt.plot(mon_points[:,0],mon_points[:,1],'or',mec='#ff0000',mfc='none')
#     plt.plot(gridLocations[:,0], gridLocations[:,1],'.k')
#     plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#     mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#     indicator=Indicator(mon)
#     sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
#     grid_points = sparse_noise._generate_grid_points_sequence()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#     mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#     indicator=Indicator(mon)
#     sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
#     sparse_noise.generate_frames()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon = Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#     frame = get_warped_square(mon.deg_coord_x,mon.deg_coord_y,(20.,25.),4.,4.,0.,foreground_color=1,background_color=0)
#     plt.imshow(frame,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
#     plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#     mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#     indicator=Indicator(mon)
#     sparse_noise=SparseNoise(mon,indicator)
#     ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
#     ds.set_stim(sparse_noise)
#     ds.trigger_display()
#     plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=20)
     indicator=Indicator(mon)
     KS_stim_all_dir=KSstimAllDir(mon,indicator,step_width=0.3)
     ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter = 2,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
     ds.set_stim(KS_stim_all_dir)
     ds.trigger_display()
     plt.show()
    #==============================================================================================================================

    #==============================================================================================================================
#     mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=5)
#     indicator=Indicator(mon)
#    
#     grating = get_grating(mon.deg_coord_x, mon.deg_coord_y, ori=0., spatial_freq=0.1, center=(60.,0.), contrast=1)
#     print grating.max()
#     print grating.min()
#     plt.imshow(grating,cmap='gray',interpolation='nearest',vmin=0., vmax=1.)
#     plt.show()
#    
#     drifting_grating = DriftingGratingCircle(mon,indicator, sf_list=(0.08,0.16),
#                                              tf_list=(4.,8.), dire_list=(0.,0.1),
#                                              con_list=(0.5,1.), size_list=(5.,10.),)
#     print '\n'.join([str(cond) for cond in drifting_grating._generate_all_conditions()])
#    
#     drifting_grating2 = DriftingGratingCircle(mon,indicator,
#                                               center=(60.,0.),
#                                               sf_list=[0.08, 0.16],
#                                               tf_list=[4.,2.],
#                                               dire_list=[np.pi/6],
#                                               con_list=[1.,0.5],
#                                               size_list=[40.],
#                                               block_dur=2.,
#                                               pregap_dur=2.,
#                                               postgap_dur=3.,
#                                               midgap_dur=1.)
#     frames =  drifting_grating2.generate_frames()
#     print '\n'.join([str(frame) for frame in frames])
#    
#     ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter = 2,is_triggered=False,is_sync_pulse_pulse=False,is_interpolate=False,display_screen=1)
#     ds.set_stim(drifting_grating2)
#     ds.trigger_display()
#     plt.show()
    #==============================================================================================================================

    # ==============================================================================================================================
#     mon=Monitor(resolution=(1200, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#     indicator=Indicator(mon)
#     uniform_contrast = UniformContrast(mon,indicator, duration=10., color=0.)
#     ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter=2,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
#     ds.set_stim(uniform_contrast)
#     ds.trigger_display()
#     plt.show()
    # ==============================================================================================================================

    # ==============================================================================================================================
#    mon = Monitor(resolution=(1080, 1920), dis=13.5, mon_width_cm=88.8, mon_height_cm=50.1, C2T_cm=33.1, C2A_cm=46.4, mon_tilt=16.22,
#                  downsample_rate=5)
#    indicator = Indicator(mon)
#    drifting_grating2 = DriftingGratingCircle(mon, indicator,
#                                              center=(60., 0.),
#                                              sf_list=[0.08],
#                                              tf_list=[4.],
#                                              dire_list=np.arange(0, 2 * np.pi, np.pi / 4),
#                                              con_list=[1.],
#                                              size_list=[20.],
#                                              block_dur=2.,
#                                              pregap_dur=2.,
#                                              postgap_dur=3.,
#                                              midgap_dur=1.)
#
#    ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, display_iter=1, is_triggered=True, is_sync_pulse_pulse=True,
#                         is_interpolate=False,display_screen=1)
#    ds.set_stim(drifting_grating2)
#    ds.trigger_display()
#
#    phases = drifting_grating2._generate_phase_list(4.)
#    print phases
    # ==============================================================================================================================
#    print 'blah'