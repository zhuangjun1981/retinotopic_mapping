
'''
Contains various stimulus routines

'''
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


def get_warped_probes(deg_coord_alt, deg_coord_azi, probes, width,
                      height, ori=0., background_color=0.):
    """
    Generate a frame (matrix) with multiple probes defined by 'porbes', `width`,
    `height` and orientation in degrees. visual degree coordinate of each pixel is
    defined by deg_coord_azi, and deg_coord_alt

    Parameters
    ----------
    deg_coord_alt : ndarray
        2d array of warped altitude coordinates of monitor pixels
    deg_coord_alt : ndarray
        2d array of warped azimuth coordinates of monitor pixels
    probes : tuple or list
        each element of probes represents a single probe (center_alt, center_azi, sign)
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

    frame = np.ones(deg_coord_azi.shape, dtype=np.float32) * background_color

    # if ori < 0. or ori > 180.:
    #      raise ValueError, 'ori should be between 0 and 180.'

    ori_arc = (ori % 360.) * 2 * np.pi / 360.

    for probe in probes:

        dis_width = np.abs(np.cos(ori_arc) * (deg_coord_azi - probe[1]) +
                           np.sin(ori_arc) * (deg_coord_alt - probe[0]))

        dis_height = np.abs(np.cos(ori_arc + np.pi / 2) * (deg_coord_azi - probe[1]) +
                            np.sin(ori_arc + np.pi / 2) * (deg_coord_alt - probe[0]))

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # fig1 = ax1.imshow(dis_width)
        # ax1.set_title('width')
        # f.colorbar(fig1, ax=ax1)
        # fig2 = ax2.imshow(dis_height)
        # ax2.set_title('height')
        # f.colorbar(fig2, ax=ax2)
        # plt.show()

        frame[np.logical_and(dis_width<=width/2.,
                             dis_height<=height/2.)] = probe[2]

    return frame


def get_circle_mask(map_alt, map_azi, center, radius):
    """
    Generate a binary mask of a circle with given `center` and `radius`

    The binary mask is generated on a map with coordinates for each pixel
    defined by `map_x` and `map_y`

    Parameters
    ----------
    map_alt  : ndarray
        altitude coordinates for each pixel on a map
    map_azi  : ndarray
        azimuth coordinates for each pixel on a map
    center : tuple
        coordinates (altitude, azimuth) of the center of the binary circle mask
    radius : float
        radius of the binary circle mask

    Returns
    -------
    circle_mask :
        binary circle mask, takes values in [0.,1.]
    """

    if map_alt.shape != map_azi.shape:
         raise ValueError, 'map_alt and map_azi should have same shape!'

    if len(map_alt.shape) != 2:
         raise ValueError, 'map_alt and map_azi should be 2-d!!'

    circle_mask = np.zeros(map_alt.shape, dtype = np.uint8)
    for (i, j), value in  np.ndenumerate(circle_mask):
        alt = map_alt[i, j]
        azi = map_azi[i, j]
        if ia.distance((alt, azi), center) <= radius:
            circle_mask[i,j] = 1
    # plt.imshow(circle_mask)
    # plt.show()
    return circle_mask


def get_grating(alt_map, azi_map, dire=0., spatial_freq=0.1,
                center=(0.,60.), phase=0., contrast=1.):
    """
    Generate a grating frame with defined spatial frequency, center location,
    phase and contrast

    Parameters
    ----------
    azi_map : ndarray
        x coordinates for each pixel on a map
    alt_map : ndarray
        y coordinates for each pixel on a map
    dire : float, optional
        orientation angle of the grating in degrees, defaults to 0.
    spatial_freq : float, optional
        spatial frequency (cycle per unit), defaults to 0.1
    center : tuple, optional
        center coordinates of circle {alt, azi}
    phase : float, optional
        defaults to 0.
    contrast : float, optional
        defines contrast. takes values in [0., 1.], defaults to 1.

    Returns
    -------
    frame :
        a frame as floating point 2-d array with grating, value range [0., 1.]
    """

    if azi_map.shape != alt_map.shape:
        raise ValueError, 'map_alt and map_azi should have same shape!'

    if len(azi_map.shape) != 2:
        raise ValueError, 'map_alt and map_azi should be 2-d!!'

    axis_arc = ((dire + 90.) * np.pi / 180.) % (2 * np.pi)

    map_azi_h = np.array(azi_map, dtype = np.float32)
    map_alt_h = np.array(alt_map, dtype = np.float32)

    distance = (np.sin(axis_arc) * (map_azi_h - center[1]) -
                np.cos(axis_arc) * (map_alt_h - center[0]))

    grating = np.sin(distance * 2 * np.pi * spatial_freq - phase)

    grating = grating * contrast  # adjust contrast

    grating = (grating + 1.) / 2. # change the scale of grating to be [0., 1.]

    return grating


# def get_sparse_loc_num_per_frame(min_alt, max_alt, min_azi, max_azi, minimum_dis):
#     """
#     given the subregion of visual space and the minmum distance between the probes
#     within a frame (definition of sparseness), return generously how many probes
#     will be presented of a given frame
#
#     Parameters
#     ----------
#     min_alt : float
#         minimum altitude of display region, in visual degrees
#     max_alt : float
#         maximum altitude of display region, in visual degrees
#     min_azi : float
#         minimum azimuth of display region, in visual degrees
#     max_azi : float
#         maximum azimuth of display region, in visual degrees
#     minimum_dis : float
#         minimum distance allowed among probes within a frame
#
#     returns
#     -------
#     probe_num_per_frame : uint
#         generously how many probes will be presented in a given frame
#     """
#     if min_alt >= max_alt:
#         raise ValueError('min_alt should be less than max_alt.')
#
#     if min_azi >= max_azi:
#         raise ValueError('min_azi should be less than max_azi.')
#
#     min_alt = float(min_alt)
#     max_alt = float(max_alt)
#     min_azi = float(min_azi)
#     max_azi = float(max_azi)
#
#     area_tot = (max_alt - min_alt) * (max_azi - min_azi)
#     area_circle = np.pi * (minimum_dis ** 2)
#     probe_num_per_frame = int(np.ceil((2.0 * (area_tot / area_circle))))
#     return probe_num_per_frame


def get_grid_locations(subregion, grid_space, monitor_azi, monitor_alt, is_include_edge=True,
                       is_plot=False):
    """
    generate all the grid points in display area (covered by both subregion and
    monitor span), designed for SparseNoise and LocallySparseNoise stimuli.

    Parameters
    ----------
    subregion : list, tuple or np.array
        the region on the monitor that will display the sparse noise,
        [min_alt, max_alt, min_azi, max_azi], all floats
    grid_space : tuple or list of two floats
        grid size of probes to be displayed, [altitude, azimuth]
    monitor_azi : 2-d array
        array mapping monitor pixels to azimuth in visual space
    monitor_alt : 2-d array
        array mapping monitor pixels to altitude in visual space
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    is_plot : bool

    Returns
    -------
    grid_locations : n x 2 array,
        refined [alt, azi] pairs of probe centers going to be displayed
    """

    rows = np.arange(subregion[0],
                     subregion[1] + grid_space[0],
                     grid_space[0])
    columns = np.arange(subregion[2],
                        subregion[3] + grid_space[1],
                        grid_space[1])

    azis, alts = np.meshgrid(columns, rows)

    grid_locations = np.transpose(np.array([alts.flatten(), azis.flatten()]))

    left_alt = monitor_alt[:, 0]
    right_alt = monitor_alt[:, -1]
    top_azi = monitor_azi[0, :]
    bottom_azi = monitor_azi[-1, :]

    left_azi = monitor_azi[:, 0]
    right_azi = monitor_azi[:, -1]
    top_alt = monitor_alt[0, :]
    bottom_alt = monitor_alt[-1, :]

    left_azi_e = left_azi - grid_space[1]
    right_azi_e = right_azi + grid_space[1]
    top_alt_e = top_alt + grid_space[0]
    bottom_alt_e = bottom_alt - grid_space[0]

    all_alt = np.concatenate((left_alt, right_alt, top_alt, bottom_alt))
    all_azi = np.concatenate((left_azi, right_azi, top_azi, bottom_azi))

    all_alt_e = np.concatenate((left_alt, right_alt, top_alt_e, bottom_alt_e))
    all_azi_e = np.concatenate((left_azi_e, right_azi_e, top_azi, bottom_azi))

    monitorPoints = np.array([all_alt, all_azi]).transpose()
    monitorPoints_e = np.array([all_alt_e, all_azi_e]).transpose()

    # get the grid points within the coverage of monitor
    if is_include_edge:
        grid_locations = grid_locations[in_hull(grid_locations, monitorPoints_e)]
    else:
        grid_locations = grid_locations[in_hull(grid_locations, monitorPoints)]

    # grid_locations = np.array([grid_locations[:, 1], grid_locations[:, 0]]).transpose()

    if is_plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(monitorPoints[:, 1], monitorPoints[:, 0], '.r', label='monitor')
        ax.plot(monitorPoints_e[:, 1], monitorPoints_e[:, 0], '.g', label='monitor_e')
        ax.plot(grid_locations[:, 1], grid_locations[:, 0], '.b', label='grid')
        ax.legend()
        plt.show()

    return grid_locations


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
        if hasattr(self, 'frame_config'):
            del self.frame_config

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
        self.frame_config = ('is_display', 'indicator color [-1., 1.]')

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

        frames = [(0, -1.)] * self.pregap_frame_num + \
                 [(1, 1.)] * displayframe_num + \
                 [(0, -1.)] * self.postgap_frame_num

        return tuple(frames)

    def _generate_frames_for_index_display(self):
        " parameters are predefined here, nothing to compute. "
        if self.indicator.is_sync:
            # Parameters that define the stimulus
            frames = ((0, -1.), (1, 1.))
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
        full_sequence = np.ones((num_frames,
                                  num_pixels_width,
                                  num_pixels_height),
                                  dtype=np.float32) * self.background

        # Compute pixel coordinates for indicator
        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel/2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel/2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel/2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel/2)

        display = self.color*np.ones((num_pixels_width,
                                      num_pixels_height),
                                      dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
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
                             dtype=np.float32)

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
                              dtype=np.float32)*self.background

        display = np.ones((np.size(self.monitor.deg_coord_x, 0),
                           np.size(self.monitor.deg_coord_x, 1)),
                           dtype=np.float32)*self.color

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
        center coordinate (altitude, azimuth) of the circle in degrees, defaults to (0.,60.).
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
    def __init__(self, monitor, indicator, coordinate='degree', center=(0., 60.),
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
        self.frame_config = ('is_display', 'indicator color [-1., 1.]')

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
                                  dtype=np.float32)

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
                                              dtype=np.float32)

        if self.coordinate == 'degree':
            map_azi = self.monitor.deg_coord_x
            map_alt = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_azi = self.monitor.lin_coord_x
            map_alt = self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        circle_mask = get_circle_mask(map_alt=map_alt, map_azi=map_azi,
                                      center=self.center, radius=self.radius).astype(np.float32)
        # plt.imshow(circle_mask)
        # plt.show()

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
                             dtype=np.float32)

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
                              dtype = np.float32)*self.background

        if self.coordinate == 'degree':
            map_azi = self.monitor.deg_coord_x
            map_alt = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_azi = self.monitor.lin_coord_x
            map_alt = self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        circle_mask = get_circle_mask(map_alt=map_alt, map_azi=map_azi,
                                      center=self.center, radius=self.radius).astype(np.float32)

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
                 grid_space=(10.,10.), probe_size=(10., 10.), probe_orientation=0.,
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
        self.frame_config = ('is_display', 'probe center (altitude, azimuth)',
                             'polarity (-1 or 1)', 'indicator color [-1., 1.]')

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

    def _get_grid_locations(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array,
            refined [alt, azi] pairs of probe centers going to be displayed
        """

        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_azi = self.monitor.deg_coord_x
            monitor_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_azi = self.monitor.lin_coord_x
            monitor_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        grid_locations = get_grid_locations(subregion=self.subregion, grid_space=self.grid_space,
                                            monitor_azi=monitor_azi, monitor_alt=monitor_alt,
                                            is_include_edge=self.is_include_edge, is_plot=is_plot)

        return grid_locations

    def _generate_grid_points_sequence(self):
        """
        generate pseudorandomized grid point sequence. if ON-OFF, consecutive
        frames should not present stimulus at same location

        Returns
        -------
        all_grid_points : list
            list of the form [grid_point, sign]
        """

        grid_points = self._get_grid_locations()

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
                  retinotopic location of the center of current square,[alt, azi]
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
            grid_points = self._get_grid_locations()
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
                    print('overlapping probes detected. ind_{}:loc{}; ind_{}:loc{}'
                          .format(i, probe_loc_0, i + 1, probe_loc_1))
                    # print ('ind_{}:loc{}'.format((i + 2) % probe_num,
                    #                              probe_locations[(i + 2) % probe_num]))
                    ind_temp = probe_ind[i + 1]
                    probe_ind[i + 1] = probe_ind[(i + 2) % probe_num]
                    probe_ind[(i + 2) % probe_num] = ind_temp
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
            if len(frames_unique) % 4 != 1:
                raise ValueError('number of frames_unique should be 4x + 1')

            index_to_display = []
            for iter in range(self.iteration):
                probe_inds = self._get_probe_index_for_one_iter_on_off(frames_unique)

                index_to_display += [0] * self.pregap_frame_num

                for probe_ind in probe_inds:
                    index_to_display += [probe_ind * 2 + 1] * probe_on_frame_num
                    index_to_display += [probe_ind * 2 + 2] * probe_off_frame_num

                index_to_display += [0] * self.postgap_frame_num

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
             coord_azi=self.monitor.deg_coord_x
             coord_alt=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_azi=self.monitor.lin_coord_x
             coord_alt=self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        full_seq = self.background * \
                   np.ones((num_unique_frames, num_pixels_width, num_pixels_height), dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
                curr_probes = ([frame[1][0], frame[1][1], frame[2]],)
                # print type(curr_probes)
                disp_mat = get_warped_probes(deg_coord_alt=coord_alt,
                                             deg_coord_azi=coord_azi,
                                             probes=curr_probes,
                                             width=self.probe_size[0],
                                             height=self.probe_size[1],
                                             ori=self.probe_orientation,
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
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

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
                           dtype=np.float32) * self.background

        for i, curr_frame in enumerate(self.frames):
            if curr_frame[0] == 1: # not a gap

                curr_probes = ([curr_frame[1][0], curr_frame[1][1], curr_frame[2]],)

                if i == 0: # first frame and (not a gap)
                    curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                      deg_coord_azi=coord_x,
                                                      probes=curr_probes,
                                                      width=self.probe_size[0],
                                                      height=self.probe_size[1],
                                                      ori=self.probe_orientation,
                                                      background_color=self.background)
                else: # (not first frame) and (not a gap)
                    if self.frames[i-1][1] is None: # (not first frame) and (not a gap) and (new square from gap)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          background_color=self.background)
                    elif (curr_frame[1]!=self.frames[i-1][1]).any() or (curr_frame[2]!=self.frames[i-1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
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
    """
    generate locally sparse noise stimulus integrates flashing indicator for
    photodiode

    This stimulus routine presents quasi-random noise in a specified region of
    the monitor. The `background` color can be customized but defaults to a
    grey value. Can specify the `subregion` of the monitor where the pixels
    will flash on and off (black and white respectively)

    Different from SparseNoise stimulus which presents only one probe at a time,
    the LocallySparseNoise presents multiple probes simultaneously to speed up
    the sampling frequency. The sparsity of probes is defined by minimum distance
    in visual degree: in any given frame, the centers of any pair of two probes
    will have distance larger than minimum distance in visual degrees. The
    method generate locally sparse noise here insures, for each iteration, all
    the locations in the subregion will be sampled once and only once.

    Parameters
    ----------
    monitor : monitor object
        contains display monitor information
    indicator : indicator object
        contains indicator information
    min_distance: float, default 20.
        the minimum distance in visual degree for any pair of probe centers
         in a given frame
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

    def __init__(self, monitor, indicator, min_distance=20., background=0., coordinate='degree',
                 grid_space=(10.,10.), probe_size=(10.,10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):

        super(LocallySparseNoise,self).__init__(monitor=monitor, indicator=indicator,
                                                background=background, coordinate = coordinate,
                                                pregap_dur=pregap_dur, postgap_dur=postgap_dur)
        """    
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        self.stim_name = 'LocallySparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.min_distance = float(min_distance)
        self.probe_orientation = probe_orientation

        self.is_include_edge = is_include_edge
        self.frame_config = ('is_display', 'probes ((altitude, azimuth, sign), ...)',
                             'iteration', 'indicator color [-1., 1.]')

        if probe_frame_num >= 2:
            self.probe_frame_num = int(probe_frame_num)
        else:
            raise ValueError('SparseNoise: probe_frame_num should be no less than 2.')

        self.is_include_edge = is_include_edge
        # self.frame_config = ('is_display', '(azimuth, altitude)',
        #                      'polarity', 'indicator_color')

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

    def _get_grid_locations(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array,
            refined [azi, alt] pairs of probe centers going to be displayed
        """


        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_azi = self.monitor.deg_coord_x
            monitor_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_azi = self.monitor.lin_coord_x
            monitor_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. Should be either "linear" or "degree".'.
                             format(self.coordinate))

        grid_locations = get_grid_locations(subregion=self.subregion, grid_space=self.grid_space,
                                            monitor_azi=monitor_azi, monitor_alt=monitor_alt,
                                            is_include_edge=self.is_include_edge, is_plot=is_plot)

        return grid_locations

    def _generate_all_probes(self):
        """
        return all possible (grid location + sign) combinations within the subregion,
        return a list of probe parameters, each element in the list is
        [center_altitude, center_azimuth, sign]
        """
        grid_locs = self._get_grid_locations()

        grid_locs = list([list(gl) for gl in grid_locs])

        if self.sign == 'ON':
            all_probes = [gl + [1.] for gl in grid_locs]
        elif self.sign == 'OFF':
            all_probes = [gl + [-1.] for gl in grid_locs]
        elif self.sign == 'ON-OFF':
            all_probes = [gl + [1.] for gl in grid_locs] + [gl + [-1.] for gl in grid_locs]
        else:
            raise ValueError('LocallySparseNoise: Cannot understand self.sign, should be '
                             'one of "ON", "OFF", "ON-OFF".')
        return all_probes

    def _generate_probe_locs_one_frame(self, probes):
        """
        given the available probes, generate a sublist of the probes for a single frame,
        all the probes in the sublist will have their visual space distance longer than
        self.min_distance. This function will also update input probes, remove the
        elements that have been selected into the sublist.

        parameters
        ----------
        probes : list of all available probes
            each elements is [center_altitude, center_azimuth, sign] for a particular probe
        min_dis : float
            minimum distance to reject probes too close to each other

        returns
        -------
        probes_one_frame : list of selected probes fo one frame
            each elements is [center_altitude, center_azimuth, sign] for a selected probe
        """

        np.random.shuffle(probes)
        probes_one_frame = []

        for probe in probes:
            is_overlap = False

            for probe_frame in probes_one_frame:
                # print probe
                # print probe_frame
                curr_dis = ia.distance([probe[0], probe[1]], [probe_frame[0], probe_frame[1]])
                if curr_dis <= self.min_distance:
                    is_overlap = True
                    break

            if not is_overlap:
                probes_one_frame.append(probe)
                probes.remove(probe)

        return probes_one_frame

    def _generate_probe_sequence_one_iteration(self, all_probes, is_redistribute=True):
        """
        given all probes to be displayed and minimum distance between any pair of two probes
        return frames of one iteration that ensure all probes will be present once

        parameters
        ----------
        all_probes : list
            all probes to be displayed, each element (center_alt, center_azi, sign). ideally
            outputs of self._generate_all_probes()
        is_redistribute : bool
            redistribute the probes among frames after initial generation or not.
            redistribute will use self._redistribute_probes() and try to minimize the difference
            of probe numbers among different frames

        returns
        -------
        frames : tuple
            each element of the frames tuple represent one display frame, the element itself
            is a tuple of the probes to be displayed in this particular frame
        """

        all_probes_cpy = list(all_probes)

        frames = []

        while len(all_probes_cpy) > 0:
            curr_frames = self._generate_probe_locs_one_frame(probes=all_probes_cpy)
            frames.append(curr_frames)

        if is_redistribute:
            frames = self._redistribute_probes(frames=frames)

        frames = tuple(tuple(f) for f in frames)

        return frames

    def _redistribute_one_probe(self, frames):

        # initiate is_moved variable
        is_moved = False

        # reorder frames from most probes to least probes
        new_frames = sorted(frames, key=lambda frame: len(frame))
        probe_num_most = len(new_frames[-1])

        # the indices of frames in new_frames that contain most probes
        frame_ind_most = []

        # the indices of frames in new_frames that contain less probes
        frame_ind_less = []

        for frame_ind, frame in enumerate(new_frames):
            if len(frame) == probe_num_most:
                frame_ind_most.append(frame_ind)
            elif len(frame) <= probe_num_most - 2:  # '-1' means well distributed
                frame_ind_less.append(frame_ind)

        # constructing a list of probes that potentially can be moved
        # each element is [(center_alt, center_azi, sign), frame_ind]
        probes_to_be_moved = []
        for frame_ind in frame_ind_most:
            frame_most = new_frames[frame_ind]
            for probe in frame_most:
                probes_to_be_moved.append((probe, frame_ind))

        # loop through probes_to_be_moved to see if any of them will fit into
        # frames with less probes, once find a case, break the loop and return
        for probe, frame_ind_src in probes_to_be_moved:
            frame_src = new_frames[frame_ind_src]
            for frame_ind_dst in frame_ind_less:
                frame_dst = new_frames[frame_ind_dst]
                if self._is_fit(probe, frame_dst):
                    frame_src.remove(probe)
                    frame_dst.append(probe)
                    is_moved = True
                    break
            if is_moved:
                break

        return is_moved, new_frames

    def _is_fit(self, probe, probes):
        """
        test if a given probe will fit a group of probes without breaking the
        sparcity

        parameters
        ----------
        probe : list or tuple of three floats
            (center_alt, center_azi, sign)
        probes : list of probes
            [(center_alt, center_zai, sign), (center_alt, center_azi, sign), ...]

        returns
        -------
        is_fit : bool
            the probe will fit or not
        """

        is_fit = True
        for probe2 in probes:
            if ia.distance([probe[0], probe[1]], [probe2[0], probe2[1]]) <= self.min_distance:
                is_fit = False
                break
        return is_fit

    def _redistribute_probes(self, frames):
        """
        attempt to redistribute probes among frames for one iteration of display
        the algorithm is to pick a probe from the frames with most probes to the
        frames with least probes and do it iteratively until it can not move
        anymore and the biggest difference of probe numbers among all frames is
        no more than 1 (most evenly distributed).

        the algorithm is implemented by self._redistribute_probes() function,
        this is just to roughly massage the probes among frames, but not the
        attempt to find the best solution.

        parameters
        ----------
        frames : list
            each element of the frames list represent one display frame, the element
            itself is a list of the probes (center_alt, center_azi, sign) to be
            displayed in this particular frame

        returns
        -------
        new_frames : list
            same structure as input frames but with redistributed probes
        """

        new_frames = list(frames)
        is_moved = True
        probe_nums = [len(frame) for frame in new_frames]
        probe_nums.sort()
        probe_diff = probe_nums[-1] - probe_nums[0]

        while is_moved and probe_diff > 1:

            is_moved, new_frames = self._redistribute_one_probe(new_frames)
            probe_nums = [len(frame) for frame in new_frames]
            probe_nums.sort()
            probe_diff = probe_nums[-1] - probe_nums[0]
        else:
            if not is_moved:
                print ('redistributing probes among frames: no more probes can be moved.')
            if probe_diff <= 1:
                print ('redistributing probes among frames: probes already well distributed.')

        return new_frames

    def _generate_frames_for_index_display(self):
        """
        compute the information that defines the frames used for index display

        parameters
        ----------
        all_probes : list
            all probes to be displayed, each element (center_alt, center_azi, sign). ideally
            outputs of self._generate_all_probes()

        returns
        -------
        frames_unique : tuple
        """
        all_probes = self._generate_all_probes()

        frames_unique = []

        gap = [0., None, None, -1.]
        frames_unique.append(gap)
        for i in range(self.iteration):
            probes_iter = self._generate_probe_sequence_one_iteration(all_probes=all_probes,
                                                                      is_redistribute=True)
            for probes in probes_iter:
                    frames_unique.append([1., probes, i, 1.])
                    frames_unique.append([1., probes, i, -1.])

        frames_unique = tuple([tuple(f) for f in frames_unique])

        return frames_unique

    def _generate_display_index(self):
        """
        compute a list of indices corresponding to each frame to display.
        """

        if self.indicator.is_sync:

            frames_unique = self._generate_frames_for_index_display()
            if len(frames_unique) % 2 == 1:
                display_num = (len(frames_unique) - 1) / 2  # number of each unique display frame
            else:
                raise ValueError('LocallySparseNoise: number of unique frames is not correct. Should be odd.')

            probe_on_frame_num = self.probe_frame_num // 2
            probe_off_frame_num = self.probe_frame_num - probe_on_frame_num

            index_to_display = []
            index_to_display += [0] * self.pregap_frame_num

            for display_ind in range(display_num):
                index_to_display += [display_ind * 2 + 1] * probe_on_frame_num
                index_to_display += [display_ind * 2 + 2] * probe_off_frame_num

            index_to_display += [0] * self.postgap_frame_num

            return frames_unique, index_to_display

        else:
            raise NotImplementedError, "method not available for non-sync indicator"

    def generate_movie_by_index(self):

        self.frames_unique, self.index_to_display = self._generate_display_index()

        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y

        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        full_seq = self.background * \
                   np.ones((num_unique_frames, num_pixels_width, num_pixels_height), dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1.:
                disp_mat = get_warped_probes(deg_coord_alt=coord_alt,
                                             deg_coord_azi=coord_azi,
                                             probes=frame[1],
                                             width=self.probe_size[0],
                                             height=self.probe_size[1],
                                             ori=self.probe_orientation,
                                             background_color=self.background)

                full_seq[i] = disp_mat

            full_seq[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[3]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        SNdict = dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        full_dict = {'stimulation': SNdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict


class DriftingGratingCircle(Stim):
    """
    Generate drifting grating circle stimulus

    Stimulus routine presents drifting grating stimulus inside
    of a circle centered at `center`. The drifting gratings are determined by
    spatial and temporal frequencies, directionality, contrast, and radius.
    The routine can generate several different gratings within
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
        coordintes for center of the stimulus (altitude, azimuth)
    sf_list : n-tuple, optional
        list of spatial frequencies in cycles/unit, defaults to `(0.08)`
    tf_list : n-tuple, optional
        list of temportal frequencies in Hz, defaults to `(4.)`
    dire_list : n-tuple, optional
        list of directions in degrees, defaults to `(0.)`
    con_list : n-tuple, optional
        list of contrasts taking values in [0.,1.], defaults to `(0.5)`
    radius_list : n-tuple
       list of radii of circles, unit defined by `self.coordinate`, defaults
       to `(10.)`
    block_dur : float, optional
        duration of each condition in seconds, defaults to `2.`
    midgap_dur : float, optional
        duration of gap between conditions, defaults to `0.5`
    iteration : int, optional
        number of times the stimulus is displayed, defaults to `1`
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 center=(0.,60.), sf_list=(0.08,), tf_list=(4.,), dire_list=(0.,),
                 con_list=(0.5,), radius_list=(10.,), block_dur=2., midgap_dur=0.5,
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
        self.sf_list = list(set(sf_list))
        self.tf_list = list(set(tf_list))
        self.dire_list = list(set(dire_list))
        self.con_list = list(set(con_list))
        self.radius_list = list(set(radius_list))

        if block_dur > 0.:
            self.block_dur = float(block_dur)
        else:
            raise ValueError('block_dur should be larger than 0 second.')

        if midgap_dur >= 0.:
            self.midgap_dur = float(midgap_dur)
        else:
            raise ValueError('midgap_dur should be no less than 0 second')

        self.iteration = iteration
        self.frame_config = ('is_display', 'isCycleStart', 'spatial frequency (cycle/deg)',
                            'temporal frequency (Hz)', 'direction (deg)',
                            'contrast [0., 1.]', 'radius (deg)', 'phase (deg)',
                             'indicator color [-1., 1.]')

        for tf in tf_list:
            period = 1. / tf
            if (0.05 * period) < (block_dur % period) < (0.95 * period):
                print period
                print block_dur % period
                print 0.95 * period
                error_msg = ('Duration of each block times tf '+ str(tf)
                             + ' should be close to a whole number!')
                raise ValueError, error_msg

    @property
    def midgap_frame_num(self):
        return int(self.midgap_dur * self.monitor.refresh_rate)

    @property
    def block_frame_num(self):
        return int(self.block_dur * self.monitor.refresh_rate)

    def _generate_all_conditions(self):
        """
        generate all possible conditions for one iteration given the lists of
        parameters

        Returns
        -------
        all_conditions : list of tuples
             all unique combinations of spatial frequency, temporal frequency,
             direction, contrast, and radius. Output depends on initialization
             parameters.

        """
        all_conditions = [(sf, tf, dire, con, size) for sf in self.sf_list
                          for tf in self.tf_list
                          for dire in self.dire_list
                          for con in self.con_list
                          for size in self.radius_list]
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

        # block_frame_num = int(self.block_dur * self.monitor.refresh_rate)

        frame_per_cycle = int(self.monitor.refresh_rate / tf)

        phases_per_cycle = list(np.arange(0, np.pi*2, np.pi*2/frame_per_cycle))

        phases = []

        while len(phases) < self.block_frame_num:
            phases += phases_per_cycle

        phases = phases[0: self.block_frame_num]
        return phases, frame_per_cycle

    @staticmethod
    def _get_ori(dire):
        """
        get orientation from direction, [0, pi)
        """
        return (dire + 90.) % 360.

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
                  contrast, [-1., 1.]
             seventh element -
                  size, float (radius of the circle in visual degree)
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
        # midgap_frames = int(self.midgap_dur*self.monitor.refresh_rate)

        for i in range(self.iteration):
            if i == 0: # very first block
                frames += [off_params for ind in range(self.pregap_frame_num)]
            else: # first block for the later iteration
                frames += [off_params for ind in range(self.midgap_frame_num)]

            all_conditions = self._generate_all_conditions()

            for j, condition in enumerate(all_conditions):
                if j != 0: # later conditions
                    frames += [off_params for ind in range(self.midgap_frame_num)]

                sf, tf, dire, con, size = condition

                # get phase list for each condition
                phases, frame_per_cycle = self._generate_phase_list(tf)
                # if (dire % 360.) >= 90. and (dire % 360. < 270.):
                #      phases = [-phase for phase in phases]

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

    def _generate_frames_for_index_display_condition(self, condi_params):
        """
        :param condi_params: list of input condition parameters, [sf, tf, dire, con, size]
                             designed for the output of self._generate_all_conditions()
        :return: frames_unique_condi: list of unique frame parameters for this particular condition
                 index_to_display_condi: list of indices of display order of the unique frames for
                                         this particular condition
        """
        phases, frame_per_cycle = self._generate_phase_list(condi_params[1])
        phases_unique = phases[0:frame_per_cycle]

        # print condi_params

        frames_unique_condi = []
        for i, ph in enumerate(phases_unique):
            if i == 0:
                frames_unique_condi.append([1, 1, condi_params[0], condi_params[1], condi_params[2],
                                            condi_params[3], condi_params[4], ph, 1.])
            else:
                frames_unique_condi.append([1, 0, condi_params[0], condi_params[1], condi_params[2],
                                            condi_params[3], condi_params[4], ph, 0.])

        index_to_display_condi = []
        while len(index_to_display_condi) < len(phases):
            index_to_display_condi += range(frame_per_cycle)
        index_to_display_condi = index_to_display_condi[0:len(phases)]

        frames_unique_condi = tuple([tuple(f) for f in frames_unique_condi])

        return frames_unique_condi, index_to_display_condi

    def _generate_frames_unique_and_condi_ind_dict(self):
        """
        compute the information that defines the frames used for index display

        :return frames_unique
                the condi_ind_in_frames_unique:
                    {
                     condi_key (same condi_key as condi_dict):
                           list of non-negative integers representing the indices of this
                           particular condition in frames_unique
                    }
        """
        if self.indicator.is_sync:
            all_conditions = self._generate_all_conditions()

            '''
            cond_dict is a dictionary constructed as following
                {
                condi_key (i.e. condi_0000):
                     {
                      'frames_unique': list of unique frame parameters for this particual condition
                                       [is_display, is_first_in_cycle, sf, tf, dire,
                                       con, size, phase, indicator_color],
                      'index_to_display': list of non-negative integers,
                     }
                }
            '''
            condi_dict = {}
            for i, condi in enumerate(all_conditions):
                frames_unique_condi, index_to_display_condi = self._generate_frames_for_index_display_condition(condi)
                condi_dict.update({'condi_{:04d}'.format(i):
                                       {'frames_unique': frames_unique_condi,
                                        'index_to_display': index_to_display_condi}
                                   })

            condi_keys = condi_dict.keys()
            condi_keys.sort()

            # handle frames_unique
            frames_unique = []
            gap_frame = (0., None, None, None, None, None, None, None, -1.)
            frames_unique.append(gap_frame)
            condi_keys.sort()
            condi_ind_in_frames_unique = {}

            for condi_key in condi_keys:
                curr_frames_unique_total = len(frames_unique)
                curr_index_to_display_condi = np.array(condi_dict[condi_key]['index_to_display'], dtype=np.uint64)
                frames_unique += list(condi_dict[condi_key]['frames_unique'])
                condi_ind_in_frames_unique.update(
                    {condi_key: list(curr_index_to_display_condi + curr_frames_unique_total)})

            return frames_unique, condi_ind_in_frames_unique
        else:
            raise NotImplementedError, "method not available for non-sync indicator"

    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """

        frames_unique, condi_ind_in_frames_unique = self._generate_frames_unique_and_condi_ind_dict()

        condi_keys = list(condi_ind_in_frames_unique.keys())

        index_to_display = []
        index_to_display += [0] * self.pregap_frame_num

        for iter in range(self.iteration):
            np.random.shuffle(condi_keys)
            for condi_ind, condi in enumerate(condi_keys):
                if iter == 0 and condi_ind == 0:
                    pass
                else:
                    index_to_display += [0] * self.midgap_frame_num
                index_to_display += condi_ind_in_frames_unique[condi]

        index_to_display += [0] * self.postgap_frame_num

        return frames_unique, index_to_display

    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames_unique, self.index_to_display = self._generate_display_index()
        # print '\n'.join([str(f) for f in self.frames_unique])

        mask_dict = self._generate_circle_mask_dict()

        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        if self.coordinate=='degree':
             coord_azi=self.monitor.deg_coord_x
             coord_alt=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_azi=self.monitor.lin_coord_x
             coord_alt=self.monitor.lin_coord_y
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
                                       dtype=np.float32)

        background_frame = self.background*np.ones((num_pixels_width,
                                                    num_pixels_height),
                                                    dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):

            if frame[0] == 1: # not a gap

                # curr_ori = self._get_ori(frame[3])

                curr_grating = get_grating(alt_map=coord_alt,
                                           azi_map=coord_azi,
                                           dire=frame[4],
                                           spatial_freq=frame[2],
                                           center=self.center,
                                           phase=frame[7],
                                           contrast=frame[5])

                curr_grating = curr_grating * 2. - 1.

                curr_circle_mask = mask_dict[frame[6]]

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
             coord_azi=self.monitor.deg_coord_x
             coord_alt=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_azi=self.monitor.lin_coord_x
             coord_alt=self.monitor.lin_coord_y

        for radius in self.radius_list:
            curr_mask = get_circle_mask(map_alt=coord_alt, map_azi=coord_azi, center=self.center, radius=radius)
            masks.update({radius: curr_mask})

        return masks

    def generate_movie(self):
        """
        Generate movie frame by frame
        """

        self.frames = self.generate_frames()
        mask_dict = self._generate_circle_mask_dict()

        if self.coordinate=='degree':
             coord_azi=self.monitor.deg_coord_x
             coord_alt=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_azi=self.monitor.lin_coord_x
             coord_alt=self.monitor.lin_coord_y
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
                       coord_azi.shape[0],
                       coord_azi.shape[1]),dtype=np.float32) * self.background
        background_frame = np.ones(coord_azi.shape,dtype=np.float32)*self.background

        for i, curr_frame in enumerate(self.frames):

            if curr_frame[0] == 1: # not a gap

                # curr_ori = self._get_ori(curr_frame[4])
                curr_grating = get_grating(alt_map=coord_alt,
                                           azi_map=coord_azi,
                                           dire= curr_frame[4],
                                           spatial_freq = curr_frame[2],
                                           center = self.center,
                                           phase = curr_frame[7],
                                           contrast = curr_frame[5])
                # plt.imshow(curr_grating)
                # plt.show()

                curr_grating = curr_grating * 2. - 1.  # change scale from [0., 1.] to [-1., 1.]

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
    """
        Generate static grating circle stimulus

        Stimulus routine presents flashing static grating stimulus inside
        of a circle centered at `center`. The static gratings are determined by
        spatial frequencies, orientation, contrast, radius and phase. The
        routine can generate several different gratings within
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
            coordintes for center of the stimulus (altitude, azimuth)
        sf_list : n-tuple, optional
            list of spatial frequencies in cycles/unit, defaults to `(0.08)`
        ori_list : n-tuple, optional
            list of directions in degrees, defaults to `(0., 90.)`
        con_list : n-tuple, optional
            list of contrasts taking values in [0.,1.], defaults to `(0.5)`
        radius_list : n-tuple, optional
           list of radii of circles, unit defined by `self.coordinate`, defaults
           to `(10.)`
        phase_list : n-tuple, optional
           list of phase of gratings in degrees, default (0., 90., 180., 270.)
        display_dur : float, optional
            duration of each condition in seconds, defaults to `0.25`
        midgap_dur, float, optional
            duration of gap between conditions, defaults to `0.`
        iteration, int, optional
            number of times the stimulus is displayed, defaults to `1`
        """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 center=(0., 60.), sf_list=(0.08,), ori_list=(0., 90.), con_list=(0.5,),
                 radius_list=(10.,), phase_list=(0., 90., 180., 270.), display_dur=0.25,
                 midgap_dur=0., iteration=1, pregap_dur=2., postgap_dur=3.):

        super(StaticGratingCircle, self).__init__(monitor=monitor,
                                                  indicator=indicator,
                                                  background=background,
                                                  coordinate=coordinate,
                                                  pregap_dur=pregap_dur,
                                                  postgap_dur=postgap_dur)
        """
        Initialize `StaticGratingCircle` stimulus object, inherits Parameters
        from `Stim` class
        """

        self.stim_name = 'DriftingGratingCircle'
        self.center = center
        self.sf_list = list(set(sf_list))
        self.phase_list = list(set([p % 360. for p in phase_list]))
        self.ori_list = list(set([o % 180. for o in ori_list]))
        self.con_list = list(set(con_list))
        self.radius_list = list(set(radius_list))

        if display_dur > 0.:
            self.display_dur = float(display_dur)
        else:
            raise ValueError('block_dur should be larger than 0 second.')

        if midgap_dur >= 0.:
            self.midgap_dur = float(midgap_dur)
        else:
            raise ValueError('midgap_dur should be no less than 0 second')

        self.iteration = iteration
        self.frame_config = ('is_display', 'spatial frequency (cycle/deg)',
                             'phase (deg)', 'orientation (deg)',
                             'contrast [0., 1.]', 'radius (deg)', 'indicator_color [-1., 1.]')

    @property
    def midgap_frame_num(self):
        return int(self.midgap_dur * self.monitor.refresh_rate)

    @property
    def display_frame_num(self):
        return int(self.display_dur * self.monitor.refresh_rate)

    @staticmethod
    def _get_dire(ori):
        return (ori + 90.) % 180.

    def _generate_circle_mask_dict(self):
        """
        generate a dictionary of circle masks for each size in size list
        """
        masks = {}
        if self.coordinate=='degree':
             coord_azi=self.monitor.deg_coord_x
             coord_alt=self.monitor.deg_coord_y
        elif self.coordinate=='linear':
             coord_azi=self.monitor.lin_coord_x
             coord_alt=self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        for radius in self.radius_list:
            curr_mask = get_circle_mask(map_alt=coord_alt, map_azi=coord_azi, center=self.center, radius=radius)
            masks.update({radius: curr_mask})

        return masks

    def _generate_all_conditions(self):
        """
        generate all possible conditions for one iteration given the lists of
        parameters

        Returns
        -------
        all_conditions : list of tuples
             all unique combinations of spatial frequency, phase,
             orientation, contrast, and radius. Output depends on initialization
             parameters.

        """
        all_conditions = [(sf, ph, ori, con, radius) for sf in self.sf_list
                          for ph in self.phase_list
                          for ori in self.ori_list
                          for con in self.con_list
                          for radius in self.radius_list]
        # random.shuffle(all_conditions)

        return all_conditions

    def _generate_frames_for_index_display(self):
        """
        generate a tuple of unique frames, each element of the tuple
        represents a unique display condition including gap

        frame structure:
            0. is_display: if gap --> 0; if display --> 1
            1. spatial frequency, cyc/deg
            2. phase, deg
            3. orientation, deg
            4. contrast, [0., 1.]
            5. radius, deg
            6. indicator color, [-1., 1.]
        """

        all_conditions =  self._generate_all_conditions()
        gap_frame = (0., None, None, None, None, None, None, None, -1.)
        frames_unique = [gap_frame]

        for condition in all_conditions:
            frames_unique.append([1, condition[0], condition[1], condition[2],
                                  condition[3], condition[4], 1.])
            frames_unique.append([1, condition[0], condition[1], condition[2],
                                  condition[3], condition[4], 0.])

        return frames_unique

    def _generate_display_index(self):

        if self.indicator.is_sync:

            display_frame_num = int(self.display_dur * self.monitor.refresh_rate)
            if display_frame_num < 2:
                raise ValueError('StaticGratingCircle: display_dur too short, should be '
                                 'at least 2 display frames.')
            indicator_on_frame_num = display_frame_num // 2
            indicator_off_frame_num = display_frame_num - indicator_on_frame_num

            frames_unique = self._generate_frames_for_index_display()
            condition_num = (len(frames_unique) - 1) / 2

            index_to_display = [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                display_sequence = range(condition_num)
                random.shuffle(display_sequence)
                for cond_ind in display_sequence:
                    index_to_display += [0] * self.midgap_frame_num
                    index_to_display += [cond_ind * 2 + 1] * indicator_on_frame_num
                    index_to_display += [cond_ind * 2 + 2] * indicator_off_frame_num

            index_to_display += [0] * self.postgap_frame_num

            # remove the extra mid gap
            index_to_display = index_to_display[self.midgap_frame_num:]

            return frames_unique, index_to_display
        else:
            raise NotImplementedError, "method not available for non-sync indicator."

    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames_unique, self.index_to_display = self._generate_display_index()

        # print '\n'.join([str(f) for f in self.frames_unique])

        mask_dict = self._generate_circle_mask_dict()

        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        mov = self.background * np.ones((num_unique_frames,
                                         num_pixels_width,
                                         num_pixels_height),
                                        dtype=np.float32)

        background_frame = self.background * np.ones((num_pixels_width,
                                                      num_pixels_height),
                                                     dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):

            if frame[0] == 1:  # not a gap

                # curr_ori = self._get_ori(frame[3])

                curr_grating = get_grating(alt_map=coord_alt,
                                           azi_map=coord_azi,
                                           dire=self._get_dire(frame[3]),
                                           spatial_freq=frame[1],
                                           center=self.center,
                                           phase=frame[2],
                                           contrast=frame[4])

                curr_grating = curr_grating * 2. - 1.

                curr_circle_mask = mask_dict[frame[5]]

                mov[i] = ((curr_grating * curr_circle_mask) +
                          (background_frame * (curr_circle_mask * -1. + 1.)))

            # add sync square for photodiode
            mov[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[-1]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        self_dict = dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')
        log = {'stimulation': self_dict,
               'monitor': mondict,
               'indicator': indicator_dict}

        return mov, log


class NaturalScene(Stim):
    #todo: finish this class
    pass


class StimulusSeparator(Stim):
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
                           dtype = np.float32)
        squareV = -1 * squareV

        stepV = np.arange(self.square_center[0] - (2*neg_x + 0.5)*self.square_size,
                          self.square_center[0] + (2*pos_x - 0.5)*self.square_size,
                          self.square_size*2)

        for i in range(len(stepV)):
            squareV[np.where(np.logical_and(map_x >= stepV[i],
                                            map_x < (stepV[i] +
                                                     self.square_size)))] = 1.0

        squareH = np.ones((np.size(map_y, 0),
                           np.size(map_y, 1)), dtype = np.float32)
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
                               np.size(map_x, 1)), dtype = np.float32)
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
                               np.size(map_y, 1)), dtype = np.float32)
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
                                                         dtype=np.float32)

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
                                 dtype = np.float32) * self.background

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

        self.stim_name = 'KSstimAllDir'
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