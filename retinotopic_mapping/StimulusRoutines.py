# -*- coding: utf-8 -*-
"""
Contains various stimulus routines

"""
import numpy as np
import matplotlib.pyplot as plt
import random

from tools import ImageAnalysis as ia


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


class Stim(object):
    """
    generic class for visual stimulation. parent class for individual
    stimulus routines.

    Parameters
    ----------
    monitor : monitor object
         the monitor used to display stimulus in the experiment
    indicator : indicator object
         the indicator used during stimulus
    background : float, optional
        background color of the monitor screen when stimulus is not being
        presented, takes values in [-1,1] and defaults to `0.` (grey)
    coordinate : str {'degree', 'linear'}, optional
        determines the representation of pixel coordinates on monitor,
        defaults to 'degree'
    pregap_dur : float, optional
        duration of gap period before stimulus, measured in seconds, defaults
        to `2.`
    postgap_dur : float, optional
        duration of gap period after stimulus, measured in seconds, defaults
        to `3.`
    """
    def __init__(self, monitor, indicator, background = 0., coordinate = 'degree',
                 pregap_dur = 2., postgap_dur = 3.):
        """
        Initialize visual stimulus object
        """

        self.monitor = monitor
        self.indicator = indicator
        self.background = background
        self.coordinate = coordinate

        if pregap_dur >= 0.:
            self.pregap_dur = pregap_dur
        else:
            raise ValueError('pregap_dur should be no less than 0.')

        if postgap_dur >= 0.:
            self.postgap_dur = postgap_dur
        else:
            raise ValueError('postgap_dur should be no less than 0.')

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
        place holder of function 'generate_movie' for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'

    def _generate_frames_for_index_display(self):
        """
        place holder of function _generate_frames_for_index_display()
        for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'

    def _generate_display_index(self):
        """
        place holder of function _generate_display_index()
        for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'

    def generate_movie_by_index(self):
        """
        place holder of function _generate_movie_by_index()
        for each specific stimulus
        """
        print 'Nothing executed! This is a place holder function'
        print 'See documentation in the respective stimulus'

    def clear(self):
        if hasattr(self, 'frames'):
            del self.frames
        if hasattr(self, 'frames_unique'):
            del self.frames_unique
        if hasattr(self, 'index_to_display'):
            del self.index_to_display

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

    Parameters
    ----------
    monitor : monitor object
        inherited monitor object from `Stim` class
    indicator : indicator object
        inherited indicator object from `Stim` class
    duration : int
        amount of time (in seconds) the stimulus is presented
    color : float, optional
        the choice of color to display in the stimulus, defaults to `0.` which
        is grey
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    background : float, optional
        color during pre and post gap, defaults to `0.` which is grey
    """
    def __init__(self, monitor, indicator, duration, color=0., pregap_dur=2.,
                 postgap_dur=3., background=0., coordinate='degree'):
        """
        Initialize UniformContrast object
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
        self.frame_config = ('is_display', 'indicator_color')

    def generate_frames(self):
        """
        generate a tuple of parameters with information for each frame.

        Information contained in each frame:
             first element -
                  during display frames, value takes on 1 and value
                  is 0 otherwise
             second element - color of indicator
                  during display value is equal to 1 and during gaps value is
                  equal to -1
        """

        displayframe_num = int(self.duration * self.monitor.refresh_rate)

        frames = [(0., -1.)] * self.pregap_frame_num + \
                 [(1., 1.)] * displayframe_num + \
                 [(0., -1.)] * self.postgap_frame_num

        return tuple(frames)
    
    def _generate_frames_for_index_display(self):
        " parameters are predefined here, nothing to compute. "
        if self.indicator.is_sync:
            # Parameters that define the stimulus
            frames = ((0.,-1.), (1.,1.))
            return frames
        else:
            raise NotImplementedError, "method not avaialable for non-sync indicator"
    
    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """
        displayframe_num = int(self.duration * self.monitor.refresh_rate)
        index_to_display = [0] * self.pregap_frame_num + [1] * displayframe_num + \
                           [0] * self.postgap_frame_num
        return index_to_display

    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames_unique = self._generate_frames_for_index_display()
        self.index_to_display = self._generate_display_index()
        
        num_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        # Initialize numpy array of 0's as placeholder for stimulus routine
        full_sequence = np.zeros((num_frames,
                                  num_pixels_width,
                                  num_pixels_height),
                                  dtype=np.float16)
        
        # Compute pixel coordinates for indicator
        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel/2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel/2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel/2)
        
        background = self.background*np.ones((num_pixels_width,
                                              num_pixels_height),
                                              dtype=np.float16)
        
        display = self.color*np.ones((num_pixels_width,
                                      num_pixels_height),
                                      dtype=np.float16)
        
        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 0:
                full_sequence[i] = background
            else:
                full_sequence[i] = display
            
            # Insert indicator pixels 
            full_sequence[i, indicator_height_min:indicator_height_max,
                          indicator_width_min:indicator_width_max] = frame[1]
            
        monitor_dict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        NF_dict = dict(self.__dict__)
        NF_dict.pop('monitor')
        NF_dict.pop('indicator')
        full_dict = {'stimulation' : NF_dict,
                     'monitor' : monitor_dict,
                     'indicator' : indicator_dict}

        return full_sequence, full_dict

    def generate_movie(self):
        """
        generate movie for uniform contrast display frame by frame.

        Returns
        -------
        full_seq : nd array, uint8
            3-d array of the stimulus to be displayed. 
        full_dict : dict
            dictionary containing the information of the stimulus.
        """

        self.frames = self.generate_frames()

        full_seq = np.zeros((len(self.frames),
                             self.monitor.deg_coord_x.shape[0],
                             self.monitor.deg_coord_x.shape[1]),
                             dtype=np.float16)

        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)

        background = np.ones((np.size(self.monitor.deg_coord_x, 0),
                              np.size(self.monitor.deg_coord_x, 1)),
                              dtype=np.float16)*self.background

        display = np.ones((np.size(self.monitor.deg_coord_x, 0),
                           np.size(self.monitor.deg_coord_x, 1)),
                           dtype=np.float16)*self.color

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
    Generate flashing circle stimulus.

    Stimulus routine presents a circle centered at the position `center`
    with given `radius`.

    Parameters
    ----------
    monitor : monitor object
        contains display monitor information.
    indicator : indicator object
        contains indicator information.
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'.
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white.
    stim_name : str
        Name of the stimulus.
    center : 2-tuple, optional
        center coordinate of the circle in degrees, defaults to `(90.,10.)`.
    radius : float, optional
        radius of the circle, defaults to `10.`
    color : float, optional
        color of the circle, takes values in [-1,1], defaults to `-1.`
    iteration : int, optional
        total number of flashes, defaults to `1`.
    flash_frame : int, optional
        number of frames that circle is displayed during each presentation
        of the stimulus, defaults to `3`.
    """
    def __init__(self, monitor, indicator, coordinate='degree', center=(90., 10.),
                 radius=10., color=-1., flash_frame_num=3, pregap_dur=2.,
                 postgap_dur=3., background=0.):

        """
        Initialize `FlashingCircle` stimulus object.
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
        self.flash_frame_num = flash_frame_num
        self.frame_config = ('is_display', 'indicator_color')

        if self.pregap_frame_num + self.postgap_frame_num == 0:
            raise ValueError('pregap_frame_num + postgap_frame_num should be larger than 0.')

        self.clear()

    def set_flash_frame_num(self, flash_frame_num):
        self.flash_frame_num = flash_frame_num
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
       

        Information contained in each frame:
           first element :
                during a gap, the value is equal to 0 and during display the
                value is equal to 1
           second element :
                corresponds to the color of indicator
                if indicator.is_sync is True, during stimulus the value is
                equal to 1., whereas during a gap the value isequal to -1.;
                if indicator.is_sync is False, indicator color will alternate
                between 1. and -1. at the frequency as indicator.freq
        Returns
        -------
        frames : list
            list of information defining each frame.
        """

        # number of frames for one round of stimulus
        iteration_frame_num = (self.pregap_frame_num +
                               self.flash_frame_num + self.postgap_frame_num)

        frames = np.zeros((iteration_frame_num,2)).astype(np.int16)

        #initilize indicator color
        frames[:,1] = -1

        for i in xrange(frames.shape[0]):

            # mark display frame and synchronized indicator
            frames[self.pregap_frame_num: self.pregap_frame_num + self.flash_frame_num, 0] = 1

            if self.indicator.is_sync:
                frames[self.pregap_frame_num: self.pregap_frame_num + self.flash_frame_num, 1] = 1

            # mark unsynchronized indicator
            if not(self.indicator.is_sync):
                if np.floor(i // self.indicator.frame_num) % 2 == 0:
                    frames[i, 1] = 1
                else:
                    frames[i, 1] = -1

        frames = [tuple(x) for x in frames]

        return tuple(frames)
    
    def _generate_frames_for_index_display(self):
        """
        frame structure: first element: is_gap (0:False; 1:True).
                         second element: indicator color [-1., 1.]
        """
        if self.indicator.is_sync:
            gap = (0., -1.)
            flash = (1., 1.)
            frames = (gap, flash)
            return frames
        else: 
            raise NotImplementedError, "method not available for non-sync indicator"
        
    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """
        index_to_display = [0] * self.pregap_frame_num + [1] * self.flash_frame_num + \
                           [0] * self.postgap_frame_num
        return index_to_display
        
    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        
        # compute unique frame parameters
        self.frames_unique = self._generate_frames_for_index_display()
        self.index_to_display = self._generate_display_index()

        num_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]
        
        full_sequence = np.zeros((num_frames,
                                  num_pixels_width,
                                  num_pixels_height), 
                                  dtype=np.float16)
        
        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel/2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel/2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)
        
        background = self.background*np.ones((num_pixels_width,
                                              num_pixels_height), 
                                              dtype=np.float16)
        
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
        
        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
                full_sequence[i] = self.color*circle_mask - background*(circle_mask-1)
                                
            full_sequence[i, indicator_height_min:indicator_height_max,
                          indicator_width_min:indicator_width_max] = frame[1]
        
        mondict=dict(self.monitor.__dict__)
        indicator_dict=dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict=dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        full_dict={'stimulation':NFdict,
                   'monitor':mondict,
                   'indicator':indicator_dict}

        return full_sequence, full_dict

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
                           indicator_width_min:indicator_width_max] = curr_frame[1]

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

    This stimulus routine presents quasi-random noise in a specified region of
    the monitor. The `background` color can be customized but defaults to a
    grey value. Can specify the `subregion` of the monitor where the pixels
    will flash on and off (black and white respectively)

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
        list or tuple, [min_alt, max_alt, min_azi, max_azi]
    sign : {'ON-OFF', 'ON', 'OFF'}, optional
        determines which pixels appear in the `subregion`, defaults to
        `'ON-Off'` so that both on and off pixels appear. If `'ON` selected
        only on pixels (white) are displayed in the noise `subregion while if
        `'OFF'` is selected only off (black) pixels are displayed in the noise
    iteration : int, optional
        number of times to present stimulus, defaults to `1`
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 grid_space=(10.,10.), probe_size=(10.,10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):

        super(SparseNoise,self).__init__(monitor=monitor,
                                         indicator=indicator,
                                         background=background,
                                         coordinate = coordinate,
                                         pregap_dur=pregap_dur,
                                         postgap_dur=postgap_dur)
        """    
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        self.stim_name = 'SparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.probe_orientation = probe_orientation

        if probe_frame_num >= 2.:
            self.probe_frame_num = int(probe_frame_num)
        else:
            raise ValueError('SparseNoise: probe_frame_num should be no less than 2.')

        self.is_include_edge = is_include_edge
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
        if iteration >= 1:
            self.iteration = int(iteration)
        else:
            raise ValueError('iteration should be no less than 1.')

        self.clear()

    def _get_grid_points(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array, 
            refined [azi, alt] pairs of probe centers going to be displayed
        """

        rows = np.arange(self.subregion[0], 
                         self.subregion[1] + self.grid_space[0], 
                         self.grid_space[0])
        columns = np.arange(self.subregion[2], 
                            self.subregion[3] + self.grid_space[1], 
                            self.grid_space[1])

        xx, yy = np.meshgrid(columns, rows)

        gridPoints = np.transpose(np.array([xx.flatten(), yy.flatten()]))

        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_x = self.monitor.deg_coord_x
            monitor_y = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_x = self.monitor.lin_coord_x
            monitor_y = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. Should be either "linear" or "degree".'.
                             format(self.coordinate))

        left_alt = monitor_y[:, 0]
        right_alt = monitor_y[:, -1]
        top_azi = monitor_x[0, :]
        bottom_azi = monitor_x[-1, :]

        left_azi = monitor_x[:, 0]
        right_azi = monitor_x[:, -1]
        top_alt = monitor_y[0, :]
        bottom_alt = monitor_y[-1, :]

        left_azi_e = left_azi - self.grid_space[1]
        right_azi_e = right_azi + self.grid_space[1]
        top_alt_e = top_alt + self.grid_space[0]
        bottom_alt_e = bottom_alt - self.grid_space[0]

        all_alt = np.concatenate((left_alt, right_alt, top_alt, bottom_alt))
        all_azi = np.concatenate((left_azi, right_azi, top_azi, bottom_azi))

        all_alt_e = np.concatenate((left_alt, right_alt, top_alt_e, bottom_alt_e))
        all_azi_e = np.concatenate((left_azi_e, right_azi_e, top_azi, bottom_azi))

        monitorPoints = np.array([all_azi, all_alt]).transpose()
        monitorPoints_e = np.array([all_azi_e, all_alt_e]).transpose()

        # get the grid points within the coverage of monitor
        if self.is_include_edge:
            gridPoints = gridPoints[in_hull(gridPoints, monitorPoints_e)]
        else:
            gridPoints = gridPoints[in_hull(gridPoints, monitorPoints)]

        if is_plot:
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(monitorPoints[:, 0], monitorPoints[:, 1], '.r', label='monitor')
            ax.plot(monitorPoints_e[:, 0], monitorPoints_e[:, 1], '.g', label='monitor_e')
            ax.plot(gridPoints[:, 0], gridPoints[:, 1], '.b', label='grid')
            ax.legend()
            plt.show()

        return gridPoints

    def _generate_grid_points_sequence(self):
        """
        generate pseudorandomized grid point sequence. if ON-OFF, consecutive
        frames should not present stimulus at same location

        Returns
        -------
        all_grid_points : list
            list of the form [grid_point, sign]
        """

        grid_points = self._get_grid_points()

        if self.sign == 'ON':
            grid_points = [[x,1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'OFF':
            grid_points = [[x,-1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'ON-OFF':
            all_grid_points = [[x,1] for x in grid_points] + [[x,-1] for x in grid_points]
            random.shuffle(all_grid_points)
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
        function to generate all the frames needed for SparseNoise stimulus

        returns a list of information of all frames as a list of tuples

        Information contained in each frame:
             first element - int
                  when stimulus is displayed value is equal to 1, otherwise
                  equal to 0,
             second element - tuple,
                  retinotopic location of the center of current square,[azi,alt]
             third element -
                  polarity of current square, 1 -> bright, -1-> dark
             forth element - color of indicator
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
                 frames += [[0., None, None, -1.]] * self.pregap_frame_num

            iter_grid_points = self._generate_grid_points_sequence()

            for grid_point in iter_grid_points:
                frames += [[1., grid_point[0], grid_point[1], 1.]] * indicator_on_frame
                frames += [[1., grid_point[0], grid_point[1], -1.]] * indicator_off_frame

            if self.postgap_frame_num>0:
                 frames += [[0., None, None, -1.]] * self.postgap_frame_num

        if self.indicator.is_sync == False:
            indicator_frame = self.indicator.frame_num
            for m in range(len(frames)):
                if np.floor(m // indicator_frame) % 2 == 0:
                    frames[m][3] = 1.
                else:
                    frames[m][3] = -1.

        frames = [tuple(x) for x in frames]

        return tuple(frames)
    
    def _generate_frames_for_index_display(self):
        """ compute the information that defines the frames used for index display"""
        if self.indicator.is_sync:
            frames_unique = []
            
            gap = [0., None, None, -1.]
            frames_unique.append(gap)
            grid_points = self._get_grid_points()
            for grid_point in grid_points:
                if self.sign == 'ON':
                    frames_unique.append([1., grid_point, 1., 1.])
                    frames_unique.append([1., grid_point, 1., -1.])
                elif self.sign == 'OFF':
                    frames_unique.append([1., grid_point, -1., 1.])
                    frames_unique.append([1., grid_point, -1., -1])
                elif self.sign == 'ON-OFF':
                    frames_unique.append([1., grid_point, 1., 1.])
                    frames_unique.append([1., grid_point, 1., -1.])
                    frames_unique.append([1., grid_point, -1., 1.])
                    frames_unique.append([1., grid_point, -1., -1])
                else:
                    raise ValueError('SparseNoise: Do not understand "sign", should '
                                     'be one of "ON", "OFF" and "ON-OFF".')

            return frames_unique
        else:
            raise NotImplementedError, "method not available for non-sync indicator"

    @staticmethod
    def _get_probe_index_for_one_iter_on_off(frames_unique):
        """
        get shuffled probe indices from frames_unique generated by
        self._generate_frames_for_index_display(), only for 'ON-OFF' stimulus

        the first element of frames_unique should be gap frame, the following
        frames should be [
                          (probe_i_ON, indictor_ON),
                          (probe_i_ON, indictor_OFF),
                          (probe_i_OFF, indictor_ON),
                          (probe_i_OFF, indictor_OFF),
                          ]

        it is designed such that no consecutive probes will hit the same visual
        field location

        return list of integers, indices of shuffled probe
        """

        if len(frames_unique) % 4 == 1:
            probe_num = (len(frames_unique) - 1) // 2
        else:
            raise ValueError('number of frames_unique should be 4x + 1')

        probe_locations = [f[1] for f in frames_unique[1::2]]
        probe_ind = np.arange(probe_num)
        np.random.shuffle(probe_ind)

        is_overlap = True
        while is_overlap:
            is_overlap = False
            for i in range(probe_num - 1):
                probe_loc_0 = probe_locations[probe_ind[i]]
                probe_loc_1 = probe_locations[probe_ind[i + 1]]
                if np.array_equal(probe_loc_0, probe_loc_1):
                    print('overlapping probes detected. ind_{}:loc{}; ind{}:loc{}'
                          .format(i, probe_loc_0, i + 1, probe_loc_1))
                    ind_temp = probe_ind[i + 1]
                    probe_ind[i + 1] = probe_ind[(i + 2) // probe_num]
                    probe_ind[(i + 2) // probe_num] = ind_temp
                    is_overlap = True

        return probe_ind

    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """

        frames_unique = self._generate_frames_for_index_display()
        probe_on_frame_num = self.probe_frame_num // 2
        probe_off_frame_num = self.probe_frame_num - probe_on_frame_num

        if self.sign == 'ON' or self.sign == 'OFF':

            if len(frames_unique) % 2 == 1:
                probe_num = (len(frames_unique) - 1) / 2
            else:
                raise ValueError('SparseNoise: number of unique frames is not correct. Should be odd.')

            index_to_display = []

            for iter in range(self.iteration):

                probe_sequence = np.arange(probe_num)
                np.random.shuffle(probe_sequence)
                index_to_display += [0] * self.pregap_frame_num

                for probe_ind in probe_sequence:
                    index_to_display += [probe_ind * 2 + 1] * probe_on_frame_num
                    index_to_display += [probe_ind * 2 + 2] * probe_off_frame_num

                index_to_display += [0] * self.postgap_frame_num

        elif self.sign == 'ON-OFF':
            #todo: finish this
            pass
        else:
            raise ValueError('SparseNoise: Do not understand "sign", should '
                             'be one of "ON", "OFF" and "ON-OFF".')
            
        return frames_unique, index_to_display
    
    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames_unique, self.index_to_display = self._generate_display_index()
        
        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]
        
        if self.coordinate=='degree':
             coord_x=self.monitor.deg_coord_x
             coord_y=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_x=self.monitor.lin_coord_x
             coord_y=self.monitor.lin_coord_y

        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)
        
        full_seq = self.background * \
                   np.ones((num_unique_frames, num_pixels_width, num_pixels_height), dtype=np.float16)
        
        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1.:
                disp_mat = get_warped_square(coord_x,
                                             coord_y,
                                             center=frame[1],
                                             width=self.probe_size[0],
                                             height=self.probe_size[1],
                                             ori=self.probe_orientation,
                                             foreground_color=frame[2],
                                             background_color=self.background)
                
                full_seq[i] = disp_mat
            
            full_seq[i, indicator_height_min:indicator_height_max,
                     indicator_width_min:indicator_width_max] = frame[3]
            
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

        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)

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
                                                          center=curr_frame[1],
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          foreground_color=curr_frame[2],
                                                          background_color=self.background)
                    elif (curr_frame[1]!=self.frames[i-1][1]).any() or (curr_frame[2]!=self.frames[i-1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        curr_disp_mat = get_warped_square(coord_x,
                                                          coord_y,
                                                          center=curr_frame[1],
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


class LocallySparseNoise(Stim):
    #todo: finish this class
    pass


class DriftingGratingCircle(Stim):
    """
    Generate drifting grating circle stimulus

    Stimulus routine presents drifting checkerboard grating stimulus inside
    of a circle centered at `center`. The drifting gratings are determined by
    spatial and temporal frequencies, directionality, contrast, and radius of
    the circle. The routine can generate several different gratings within
    one presentation by specifying multiple values of the parameters which
    characterize the stimulus.

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

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 center=(60.,0.), sf_list=(0.08,), tf_list=(4.,), dire_list=(0.,),
                 con_list=(0.5,), size_list=(5.,), block_dur=2., midgap_dur=0.5,
                 iteration=1, pregap_dur=2., postgap_dur=3.):

        super(DriftingGratingCircle,self).__init__(monitor=monitor,
                                                   indicator=indicator,
                                                   background=background,
                                                   coordinate=coordinate,
                                                   pregap_dur=pregap_dur,
                                                   postgap_dur=postgap_dur)
        """
        Initialize `DriftingGratingCircle` stimulus object, inherits Parameters
        from `Stim` class
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
             first element -
                  value equal to 1 during stimulus and 0 otherwise
             second element -
                  on first frame in a cycle value takes on 1, and otherwise is
                  equal to 0.
             third element -
                  spatial frequency
             forth element -
                  temporal frequency
             fifth element -
                  direction, [0, 2*pi)
             sixth element -
                  contrast
             seventh element -
                  size (radius of the circle)
             eighth element -
                  phase, [0, 2*pi)
             ninth element -
                  indicator color [-1, 1]. Value is equal to 1 on the first
                  frame of each cycle, -1 during gaps and otherwise 0.

             during gap frames the second through the eighth elements should
             be 'None'.
        """

        frames = []
        off_params = [0, None,None,None,None,None,None,None,-1.]
        midgap_frames = int(self.midgap_dur*self.monitor.refresh_rate)
        

        for i in range(self.iteration):
            if i == 0: # very first block
                frames += [off_params for ind in range(self.pregap_frame_num)]
            else: # first block for the later iteration
                frames += [off_params for ind in range(midgap_frames)]

            all_conditions = self._generate_all_conditions()

            for j, condition in enumerate(all_conditions):
                if j != 0: # later conditions
                    frames += [off_params for ind in range(midgap_frames)]

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

                    frames.append([1, first_in_cycle, sf, tf, dire,
                                   con, size, phase, float(first_in_cycle)])

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
    
    def _generate_frames_for_index_display(self):
        """ compute the information that defines the frames used for index display 
        
            First chunk - 1 frame containing an off condition
            second chunk - chunk of size depending on temporal frequency
            third chunk - 1 frame containing an off midgap cond
            fourth - chunk of size depedning on temporal freq
            ...
            
        """
        if self.indicator.is_sync:
            single_run_frames = []
            off_params = (0, None,None,None,None,None,None,None,-1.)
            midgap_frame_num = int(self.midgap_dur*self.monitor.refresh_rate)
            block_gap_num = int(self.block_dur*self.monitor.refresh_rate)
            
            # Used to store the length of a given block. 'off' blocks will always
            # be 1 whereas 'on' blocks are of variable length. we need this information
            # in order to play the stimulus correctly
            num_unique_block_frames = [1]
            
            # used to store number of repeats for each frame, e.g. pregap_frame_num
            # postgap_frame_num, block_frame_num, ...
            num_disp_iters = [self.pregap_frame_num]
            
            
            for i in range(self.iteration):
                if i!=0:
                    num_unique_block_frames.append(1)
                    num_disp_iters.append(midgap_frame_num)
                single_run_frames += [off_params]
                
                # Compute all combinations of defining parameters
                all_conditions = self._generate_all_conditions()
                
                for j, condition in enumerate(all_conditions):
                    if j!=0:
                        single_run_frames += [off_params]
                        num_unique_block_frames.append(1)
                        num_disp_iters.append(midgap_frame_num)
                    sf, tf, dire, con, size = condition
                    
                    # Compute phase list
                    phases, frame_per_cycle = self._generate_phase_list(tf)
                    
                    phases = phases[:frame_per_cycle]
                    num_unique_block_frames.append(frame_per_cycle)
                    
                    stim_on = []
                    
                    if (dire % (np.pi * 2)) >= np.pi:
                             phases = [-phase for phase in phases]
                    
                    # Make each unique frame parameters
                    for k, phase in enumerate(phases): 
        
                        # mark first frame of each cycle
                        # if phase == 0 then it is first frame of a cycle
                        if k == 0:
                            stim_on += [(1, sf, tf, dire, con, size, phase, 1.)]
                            
                        else:
                            stim_on += [(1, sf, tf, dire, con, size, phase, 0.)]
                            
                    single_run_frames += stim_on
                    num_disp_iters.append(block_gap_num)
                    
            
            single_run_frames += [off_params]
            num_unique_block_frames.append(1)
            num_disp_iters.append(self.postgap_frame_num)
                   
            return single_run_frames, num_unique_block_frames, num_disp_iters
        
        else:
            raise NotImplementedError, "method not available for non-sync indicator"
    
    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """
        (frames, 
         num_unique_block_frames, 
         num_disp_iters) = self._generate_frames_for_index_display()
        
        # Compute list of indices of each frame to display
        index_to_display = []
        
        cumsum = np.cumsum(num_unique_block_frames)
        cumsum = list(np.insert(cumsum,0,0))
        
        frame_blocks = []
        
        for i in range(len(cumsum)-1):
            frame_blocks.append(range(cumsum[i],cumsum[i+1]))
            
        for n,frame_block in enumerate(frame_blocks):
            if len(frame_block)==1:
                index_to_display += frame_block*num_disp_iters[n]
            else:
                repeats = (num_disp_iters[n] / num_unique_block_frames[n]) + 1
                
                repeated_block = frame_block*repeats
                
                repeated_block = repeated_block[:num_disp_iters[n]]
                
                index_to_display += repeated_block
                
        
        return frames, index_to_display
        
    
    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames, self.index_to_display = self._generate_display_index()
        
        mask_dict = self._generate_circle_mask_dict()
        
        num_unique_frames = len(self.frames)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]
        
        if self.coordinate=='degree':
             coord_x=self.monitor.deg_coord_x
             coord_y=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_x=self.monitor.lin_coord_x
             coord_y=self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel/2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel/2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)
        
        mov = self.background*np.ones((num_unique_frames,
                                       num_pixels_width,
                                       num_pixels_height),
                                       dtype=np.float16)

        background_frame = self.background*np.ones((num_pixels_width,
                                                    num_pixels_height),
                                                    dtype=np.float16)
        
        for i, frame in enumerate(self.frames):

            if frame[0] == 1: # not a gap
        
                curr_ori = self._get_ori(frame[3])

                curr_grating = get_grating(coord_x,
                                           coord_y,
                                           ori = curr_ori,
                                           spatial_freq = frame[1],
                                           center = self.center,
                                           phase = frame[6],
                                           contrast = frame[4])
                curr_grating = curr_grating*2. - 1.

                curr_circle_mask = mask_dict[frame[5]]

                mov[i] = ((curr_grating * curr_circle_mask) +
                             (background_frame * (curr_circle_mask * -1. + 1.)))
                    
                    

            #add sync square for photodiode
            mov[i, indicator_height_min:indicator_height_max,
                indicator_width_min:indicator_width_max] = frame[-1]
                                
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

        indicator_width_min = (self.indicator.center_width_pixel 
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel 
                               + self.indicator.width_pixel/2)
        indicator_height_min = (self.indicator.center_height_pixel 
                                - self.indicator.height_pixel/2)
        indicator_height_max = (self.indicator.center_height_pixel 
                                + self.indicator.height_pixel / 2)

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


class StaticGratingCircle(Stim):
    #todo: finish this class
    pass


class KSstim(Stim):
    """
    generate Kalatsky & Stryker stimulus

    Kalatsky & Stryker (KS) stimulus routine presents checkerboard gratings
    that drift against a fixed `background` color.

    Parameters
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
        number of frames in one flicker, defaults to `10`
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
        roughly determines speed of the drifting grating, defaults to `1`
    iteration : int, optional
        number of times that the stimulus will be repeated, defaults to `1`
    pregap_dur : float, optional
        number of seconds before stimulus is presented, defaults to `2`
    postgap_dur : float, optional
        number of seconds after stimulus is presented, defaults to `2`
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 square_size=25., square_center=(0,0), flicker_frame=10,
                 sweep_width=20., step_width=0.15, direction='B2U', sweep_frame=1,
                 iteration=1, pregap_dur=2., postgap_dur=3.):

        super(KSstim,self).__init__(monitor=monitor,
                                    indicator=indicator,
                                    coordinate=coordinate,
                                    background=background,
                                    pregap_dur=pregap_dur,
                                    postgap_dur=postgap_dur)
        """
        Initialize Kalatsky & Stryker stimulus object 
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
             first element -
                  during stimulus value is equal to 1 and 0 otherwise
             second element -
                  square polarity, 1->not reversed; -1->reversed
             third element:
                  sweeps, index in sweep table
             forth element -
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

    Generalizes the KS stimulus routine so that the drifting gratings can go
    in all four directions

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
        number of frames per flicker while stimulus is being presented,
        defaults to `6`
    sweep_width : float, optional
        width of sweeps. defaults to `20.`
    step_width : float, optional
        width of steps. defaults to `0.15`.
    sweep_frame : int, optional
        roughly determines speed of the drifting grating, defaults to `1`
    iteration : int, optional
        number of times stimulus will be presented, defaults to `1`
    pregap_dur : float, optional
        number of seconds before stimulus is presented, defaults to `2.`
    postgap_dur : float, optional
        number of seconds after stimulus is presented, defaults to `3.`
    """
    def __init__(self, monitor, indicator, coordinate='degree', background=0.,
                 square_size=25, square_center=(0,0), flicker_frame=6, sweep_width=20.,
                 step_width=0.15, sweep_frame=1, iteration=1, pregap_dur=2.,
                 postgap_dur=3.):
        """
        Initialize stimulus object
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