# -*- coding: utf-8 -*-
"""
Used to store the display monitor and particular geometry used within a given 
experimental setup. The `Monitor` class holds references to the sizing of the 
monitor that is used to display stimulus routines and contains the necessary 
geometrical description of where the subject's eye is placed with respect to the 
display monitor. The `Indicator` class, on the other hand, is generally used in 
order to gain finer scales of temporal precision. This is done  by connecting a 
photodiode indicator to one of the corners of the display monitor and ideally 
synchronising the indicator with the triggering of specific stimulus events.

The module will most definitely be used in conjunction with the `DisplayStimulus`
and `StimulusRoutines` modules.

"""
import numpy as np

class Monitor(object):
    """
    monitor object created by Jun, has the method "remap" to generate the 
    spherical corrected coordinates in degrees
    
    This object contains the relevant data for the monitor used within a 
    given experimental setup. When initialized, the rectangular coordinates
    of the pixels on the monitor are computed and stored as `lin_coord_x`, 
    `lin_coord_y`. The rectangular coordinates are then transformed and 
    warped by calling the `remap` method to populate the `deg_coord_x` and
    `deg_coord_y` attributes.
   
    Parameters
    ----------
    resolution : 2-tuple
        values of the monitor resolution.
    dis : float 
         distance from the subject's eyeball to the monitor (in cm).
    mon_width_cm : float
        width of monitor (in cm).
    mon_height_cm : float
        height of monitor (in cm).
    C2T_cm : float
        distance from the subject's gaze center to the top of the monitor
        (in cm).
    C2A_cm : float
        distance from the subject's gaze center to the anterior edge of the 
        monitor (in cm).
    mon_tilt : float
        angle between the subjec'ts body axis and the plane that the monitor 
        lies in (in degrees).
    visual_field : str from {'right','left'}, optional
        the subject's eye that is facing the monitor, defaults to 'right'.
    deg_coord_x : ndarray, optional
         array of warped x pixel coordinates, defaults to `None`.
    deg_coord_y : ndarray, optional
         array of warped y pixel coordinates, defaults to `None`.
    name : str, optional
         name of the monitor, defaults to `testMonitor`.
    gamma : optional
         for gamma correction, defaults to `None`, since as of now gamma
         correction is taken care of with psychopy. Can be customized
         if desired.
    gamma_grid : optional
         for gamme correction, defaults to `None`, since as of now gamma
         correction is taken care of with pyschopy. Can be customized
         if desired.
    luminance : optional
         the luminance of the display monitor luminance, defaults to `None`.
    downsample_rate : int, optional
         the desred rate of downsampling the monitor pixels, defaults to 10.
    refresh_rate : float, optional
        the refresh rate of the monitor (in Hz), defaults to 60.
        
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
        """ Initialize monitor object."""
                     
        if resolution[0] % downsample_rate != 0 \
                       or resolution[1] % downsample_rate != 0:           
           raise ArithmeticError, 'Resolution pixel numbers are not' \
           ' divisible by down sampling rate'
        
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
        self.C2B_cm = self.mon_height_cm - self.C2T_cm
        self.C2P_cm = self.mon_width_cm - self.C2A_cm
        
        resolution = [0,0]        
        resolution[0] = self.resolution[0]/downsample_rate
        resolution[1] = self.resolution[1]/downsample_rate
        
        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]), 
                                               range(resolution[0]))
        
        if self.visual_field == "left": 
            map_x = np.linspace(self.C2A_cm, -1.0 * self.C2P_cm, resolution[1])
            
        if self.visual_field == "right":
            map_x = np.linspace(-1 * self.C2A_cm, self.C2P_cm, resolution[1])
            
        map_y = np.linspace(self.C2T_cm, -1.0 * self.C2B_cm, resolution[0])
        old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse = False)
        
        self.lin_coord_x = old_map_x
        self.lin_coord_y = old_map_y
        
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
         
        Function is called when the monitor object is initialized.

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

    used in order to gain finer scales of temporal precision. This is done
    by connecting a photodiode indicator to one of the corners of the 
    display monitor and ideally synchronising the indicator with the 
    triggering of specific stimulus events.
    
    Parameters
    ----------
    monitor : monitor object
        The monitor used to display stimulus within the experimental setup.
    width_cm : float, optional
        width of the size of the indicator (in cm), defaults to `3.`
    height_cm : float, optional
         height of the size of the indicator (in cm), defaults to `3.`
    position : str from {'northeast','northwest','southwest','southeast'}
         the placement of the indicator, defaults to 'northeast'.
    is_sync : bool, optional
         determines whether the indicator is synchronized with the stimulus,
         defaults to `True`.
    freq : float, optional
        frequency of the photodiode, defaults to `2.`
        
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
        
        """
        
        self.monitor=monitor
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.width_pixel, self.height_pixel = self.get_size_pixel()
        self.position = position
        self.center_width_pixel, self.center_height_pixel = self.get_center()
        self.is_sync = is_sync

        if is_sync == False:
            self.freq = freq 
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
            raise LookupError, '`position` attribute not in' \
            ' {"northeast","northwest","southeast","southwest"}'

        return int(center_width), int(center_height)

    def get_frames(self):
        """
        if not synchronized with stimulation, get frame numbers of each update
        
        """

        refresh_rate = self.monitor.refresh_rate

        if refresh_rate % self.freq != 0:
            raise ArithmeticError, "`freq` not divisble by monitor ref rate."

        return refresh_rate/self.freq
