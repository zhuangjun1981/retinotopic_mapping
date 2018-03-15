"""
Contains various stimulus routines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import h5py
from tools import ImageAnalysis as ia
from tools import FileTools as ft

try:
    import skimage.external.tifffile as tf
except ImportError:
    import tifffile as tf


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
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


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
    deg_coord_azi : ndarray
        2d array of warped azimuth coordinates of monitor pixels
    probes : tuple or list
        each element of probes represents a single probe (center_alt, center_azi, sign)
    width : float
         width of the square in degrees
    height : float
         height of the square in degrees
    ori : float
        angle in degree, should be [0., 180.]
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

        frame[np.logical_and(dis_width <= width / 2.,
                             dis_height <= height / 2.)] = probe[2]

    return frame


def blur_cos(dis, sigma):
    """
    return a smoothed value [0., 1.] given the distance to center (with sign)
    and smooth width. this is using cosine curve to smooth edge

    parameters
    ----------
    dis : ndarray
        array that store the distance from the current pixel to blurred band center
    sigma : float
        definition of the width of blurred width, here is the length represent
        half cycle of the cosin function

    returns
    -------
    blurred : float
        blurred value
    """
    dis_f = dis.astype(np.float32)
    sigma_f = abs(float(sigma))

    blur_band = (np.cos((dis_f - (sigma_f / -2.)) * np.pi / sigma_f) + 1.) / 2.

    # plt.imshow(blur_band)
    # plt.show()

    blur_band[dis_f < (sigma_f / -2.)] = 1.
    blur_band[dis_f > (sigma_f / 2.)] = 0.

    # print blur_band.dtype

    return blur_band


def get_circle_mask(map_alt, map_azi, center, radius, is_smooth_edge=False,
                    blur_ratio=0.2, blur_func=blur_cos, is_plot=False):
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
    is_smooth_edge : bool
        if True, use 'blur_ratio' and 'blur_func' to smooth circle edge
    blur_ratio : float, option, default 0.2
        the ratio between blurred band width to radius, should be smaller than 1
        the middle of blurred band is the circle edge
    blur_func : function object to blur edge
    is_plot : bool

    Returns
    -------
    circle_mask : ndarray (dtype np.float32) with same shape as map_alt and map_azi
        if is_smooth_edge is True
            weighted circle mask, with smoothed edge
        if is_smooth_edge is False
            binary circle mask, takes values in [0.,1.]
    """

    if map_alt.shape != map_azi.shape:
        raise ValueError('map_alt and map_azi should have same shape.')

    if len(map_alt.shape) != 2:
        raise ValueError('map_alt and map_azi should be 2-d.')

    dis_mat = np.sqrt((map_alt - center[0]) ** 2 + (map_azi - center[1]) ** 2)
    # plt.imshow(dis_mat)
    # plt.show()

    if is_smooth_edge:
        sigma = radius * blur_ratio
        circle_mask = blur_func(dis=dis_mat - radius, sigma=sigma)
    else:
        circle_mask = np.zeros(map_alt.shape, dtype=np.float32)
        circle_mask[dis_mat <= radius] = 1.

    if is_plot:
        plt.imshow(circle_mask)
        plt.show()

    return circle_mask


def get_grating(alt_map, azi_map, dire=0., spatial_freq=0.1,
                center=(0., 60.), phase=0., contrast=1.):
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
        raise ValueError('map_alt and map_azi should have same shape.')

    if len(azi_map.shape) != 2:
        raise ValueError('map_alt and map_azi should be 2-d.')

    axis_arc = ((dire + 90.) * np.pi / 180.) % (2 * np.pi)

    map_azi_h = np.array(azi_map, dtype=np.float32)
    map_alt_h = np.array(alt_map, dtype=np.float32)

    distance = (np.sin(axis_arc) * (map_azi_h - center[1]) -
                np.cos(axis_arc) * (map_alt_h - center[0]))

    grating = np.sin(distance * 2 * np.pi * spatial_freq - phase)

    grating = grating * contrast  # adjust contrast

    grating = (grating + 1.) / 2.  # change the scale of grating to be [0., 1.]

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

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 pregap_dur=2., postgap_dur=3.):
        """
        Initialize visual stimulus object
        """

        self.monitor = monitor
        self.indicator = indicator

        if background < -1. or background > 1.:
            raise ValueError('parameter "background" should be a float within [-1., 1.].')
        else:
            self.background = float(background)

        if coordinate not in ['degree', 'linear']:
            raise ValueError('parameter "coordinate" should be either "degree" or "linear".')
        else:
            self.coordinate = coordinate

        if pregap_dur >= 0.:
            self.pregap_dur = float(pregap_dur)
        else:
            raise ValueError('pregap_dur should be no less than 0.')

        if postgap_dur >= 0.:
            self.postgap_dur = float(postgap_dur)
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
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def generate_movie(self):
        """
        place holder of function 'generate_movie' for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. '
              'See documentation in the respective stimulus. \n'
              'It is possible that full sequence generation is not'
              'implemented in this particular stimulus. Try '
              'generate_movie_by_index() function to see if indexed '
              'sequence generation is implemented.')

    def _generate_frames_for_index_display(self):
        """
        place holder of function _generate_frames_for_index_display()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def _generate_display_index(self):
        """
        place holder of function _generate_display_index()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def generate_movie_by_index(self):
        """
        place holder of function generate_movie_by_index()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def clear(self):
        if hasattr(self, 'frames'):
            del self.frames
        if hasattr(self, 'frames_unique'):
            del self.frames_unique
        if hasattr(self, 'index_to_display'):
            del self.index_to_display

        # for StaticImages
        if hasattr(self, 'images_wrapped'):
            del self.images_wrapped
        if hasattr(self, 'images_dewrapped'):
            del self.images_dewrapped
        if hasattr(self, 'altitude_wrapped'):
            del self.altitude_wrapped
        if hasattr(self, 'azimuth_wrapped'):
            del self.azimuth_wrapped
        if hasattr(self, 'altitude_dewrapped'):
            del self.altitude_dewrapped
        if hasattr(self, 'azimuth_dewrapped'):
            del self.azimuth_dewrapped

    def set_monitor(self, monitor):
        self.monitor = monitor
        self.clear()

    def set_indicator(self, indicator):
        self.indicator = indicator
        self.clear()

    def set_pregap_dur(self, pregap_dur):
        if pregap_dur >= 0.:
            self.pregap_dur = float(pregap_dur)
        else:
            raise ValueError('pregap_dur should be no less than 0.')
        self.clear()

    def set_postgap_dur(self, postgap_dur):
        if postgap_dur >= 0.:
            self.postgap_dur = float(postgap_dur)
        else:
            raise ValueError('postgap_dur should be no less than 0.')

    def set_background(self, background):
        if background < -1. or background > 1.:
            raise ValueError('parameter "background" should be a float within [-1., 1.].')
        else:
            self.background = float(background)
        self.clear()

    def set_coordinate(self, coordinate):
        if coordinate not in ['degree', 'linear']:
            raise ValueError('parameter "coordinate" should be either "degree" or "linear".')
        self.coordinate = coordinate


class UniformContrast(Stim):
    """
    Generate full field uniform luminance for recording spontaneous activity.
    Inherits from Stim.

    The full field uniform luminance stimulus presents a fixed background color
    which is normally taken to be grey.

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    color : float, optional
        the choice of color to display in the stimulus, defaults to `0.` which
        is grey
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
        self.color = float(color)
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
        """ parameters are predefined here, nothing to compute. """
        if self.indicator.is_sync:
            # Parameters that define the stimulus
            frames = ((0, -1.), (1, 1.))
            return frames
        else:
            raise NotImplementedError("method not avaialable for non-sync indicator.")

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
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        display = self.color * np.ones((num_pixels_width,num_pixels_height),
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
        full_dict = {'stimulation': NF_dict,
                     'monitor': monitor_dict,
                     'indicator': indicator_dict}

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
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        background = np.ones((np.size(self.monitor.deg_coord_x, 0),
                              np.size(self.monitor.deg_coord_x, 1)),
                             dtype=np.float32) * self.background

        display = np.ones((np.size(self.monitor.deg_coord_x, 0),
                           np.size(self.monitor.deg_coord_x, 1)),
                          dtype=np.float32) * self.color

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
                print('Generating numpy sequence: ' +
                       str(int(100 * (i + 1) / len(self.frames))) + '%')

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
        contains display monitor information
    indicator : indicator object
        contains indicator information
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    center : 2-tuple, optional
        center coordinate (altitude, azimuth) of the circle in degrees, defaults to (0.,60.).
    radius : float, optional
        radius of the circle, defaults to `10.`
    is_smooth_edge : bool
        True, smooth circle edge with smooth_width_ratio and smooth_func
        False, do not smooth edge
    smooth_width_ratio : float, should be smaller than 1.
        the ratio between smooth band width and radius, circle edge is the middle
        of smooth band
    smooth_func : function object
        this function take to inputs
            first, ndarray storing the distance from each pixel to smooth band center
            second, smooth band width
        returns smoothed mask with same shape as input ndarray
    color : float, optional
        color of the circle, takes values in [-1,1], defaults to `-1.`
    iteration : int, optional
        total number of flashes, defaults to `1`.
    flash_frame : int, optional
        number of frames that circle is displayed during each presentation
        of the stimulus, defaults to `3`.
    """

    def __init__(self, monitor, indicator, coordinate='degree', center=(0., 60.),
                 radius=10., is_smooth_edge=False, smooth_width_ratio=0.2,
                 smooth_func=blur_cos, color=-1., flash_frame_num=3,
                 pregap_dur=2., postgap_dur=3., background=0., midgap_dur=1.,
                 iteration=1):

        """
        Initialize `FlashingCircle` stimulus object.
        """

        super(FlashingCircle, self).__init__(monitor=monitor,
                                             indicator=indicator,
                                             background=background,
                                             coordinate=coordinate,
                                             pregap_dur=pregap_dur,
                                             postgap_dur=postgap_dur)

        self.stim_name = 'FlashingCircle'
        self.center = center
        self.radius = float(radius)
        self.color = float(color)
        self.flash_frame_num = int(flash_frame_num)
        self.frame_config = ('is_display', 'indicator color [-1., 1.]')
        self.is_smooth_edge = is_smooth_edge
        self.smooth_width_ratio = float(smooth_width_ratio)
        self.smooth_func = smooth_func
        self.midgap_dur = float(midgap_dur)
        self.iteration = int(iteration)

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

    @property
    def midgap_frame_num(self):
        return int(self.midgap_dur * self.monitor.refresh_rate)

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

        frames = [[0, -1.]] * self.pregap_frame_num

        for iter in range(self.iteration):

            if self.indicator.is_sync:
                frames += [[0, -1.]] * self.midgap_frame_num
                frames += [[1, 1.]] * self.flash_frame_num
            else:
                frames += [[0, -1.]] * self.midgap_frame_num
                frames += [[1, -1.]] * self.flash_frame_num

        frames += [[0, -1.]] * self.postgap_frame_num

        frames = frames[self.midgap_frame_num:]

        if not self.indicator.is_sync:
            for frame_ind in xrange(frames.shape[0]):
                # mark unsynchronized indicator
                if np.floor(frame_ind // self.indicator.frame_num) % 2 == 0:
                    frames[frame_ind, 1] = 1.
                else:
                    frames[frame_ind, 1] = -1.

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
        if self.indicator.is_sync:

            index_to_display = [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                index_to_display += [0] * self.midgap_frame_num
                index_to_display += [1] * self.flash_frame_num

            index_to_display += [0] * self.postgap_frame_num
            index_to_display = index_to_display[self.midgap_frame_num:]

            return index_to_display
        else:
            raise NotImplementedError, "method not available for non-sync indicator"

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
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        background = self.background * np.ones((num_pixels_width,
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
                                      center=self.center, radius=self.radius,
                                      is_smooth_edge=self.is_smooth_edge,
                                      blur_ratio=self.smooth_width_ratio,
                                      blur_func=self.smooth_func).astype(np.float32)
        # plt.imshow(circle_mask)
        # plt.show()

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
                full_sequence[i] = self.color * circle_mask - background * (circle_mask - 1)

            full_sequence[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[1]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        NFdict.pop('smooth_func')
        full_dict = {'stimulation': NFdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_sequence, full_dict

    def generate_movie(self):
        """
        generate movie frame by frame.
        """

        self.frames = self.generate_frames()

        full_seq = np.zeros((len(self.frames), self.monitor.deg_coord_x.shape[0],
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
                              np.size(self.monitor.deg_coord_x, 1)),
                             dtype=np.float32) * self.background

        if self.coordinate == 'degree':
            map_azi = self.monitor.deg_coord_x
            map_alt = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_azi = self.monitor.lin_coord_x
            map_alt = self.monitor.lin_coord_y
        else:
            raise LookupError, "`coordinate` not in {'linear','degree'}"

        circle_mask = get_circle_mask(map_alt=map_alt, map_azi=map_azi,
                                      center=self.center, radius=self.radius,
                                      is_smooth_edge=self.is_smooth_edge,
                                      blur_ratio=self.smooth_width_ratio,
                                      blur_func=self.smooth_func).astype(np.float32)

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

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: '
                       + str(int(100 * (i + 1) / len(self.frames))) + '%')

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        NFdict.pop('smooth_func')
        full_dict = {'stimulation': NFdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    grid_space : 2-tuple of floats, optional
        first coordinate is altitude, second coordinate is azimuth
    probe_size : 2-tuple of floats, optional
        size of flicker probes. First coordinate defines the width, and
        second coordinate defines the height
    probe_orientation : float, optional
        orientation of flicker probes
    probe_frame_num : int, optional
        number of frames for each square presentation
    subregion : list or tuple
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
                 grid_space=(10., 10.), probe_size=(10., 10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):
        """
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        super(SparseNoise, self).__init__(monitor=monitor,
                                          indicator=indicator,
                                          background=background,
                                          coordinate=coordinate,
                                          pregap_dur=pregap_dur,
                                          postgap_dur=postgap_dur)

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
            grid_points = [[x, 1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'OFF':
            grid_points = [[x, -1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'ON-OFF':
            all_grid_points = [[x, 1] for x in grid_points] + [[x, -1] for x in grid_points]
            random.shuffle(all_grid_points)
            # remove coincident hit of same location by continuous frames
            print('removing coincident hit of same location with continuous frames:')
            while True:
                iteration = 0
                coincident_hit_num = 0
                for i, grid_point in enumerate(all_grid_points[:-3]):
                    if (all_grid_points[i][0] == all_grid_points[i + 1][0]).all():
                        all_grid_points[i + 1], all_grid_points[i + 2] = all_grid_points[i + 2], all_grid_points[i + 1]
                        coincident_hit_num += 1
                iteration += 1
                print('iteration:' + iteration + '  continous hits number:' + coincident_hit_num)
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

        frames += [[0., None, None, -1.]] * self.pregap_frame_num

        for i in range(self.iteration):

            iter_grid_points = self._generate_grid_points_sequence()

            for grid_point in iter_grid_points:
                frames += [[1., grid_point[0], grid_point[1], 1.]] * indicator_on_frame
                frames += [[1., grid_point[0], grid_point[1], -1.]] * indicator_off_frame

        frames += [[0., None, None, -1.]] * self.postgap_frame_num

        if not self.indicator.is_sync:
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

            frames_unique = tuple([tuple(f) for f in frames_unique])

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
                    # print('overlapping probes detected. ind_{}:loc{}; ind_{}:loc{}'
                    #       .format(i, probe_loc_0, i + 1, probe_loc_1))
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

            index_to_display += [0] * self.pregap_frame_num

            for iter in range(self.iteration):

                probe_sequence = np.arange(probe_num)
                np.random.shuffle(probe_sequence)

                for probe_ind in probe_sequence:
                    index_to_display += [probe_ind * 2 + 1] * probe_on_frame_num
                    index_to_display += [probe_ind * 2 + 2] * probe_off_frame_num

            index_to_display += [0] * self.postgap_frame_num

        elif self.sign == 'ON-OFF':
            if len(frames_unique) % 4 != 1:
                raise ValueError('number of frames_unique should be 4x + 1')

            index_to_display = []

            index_to_display += [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                probe_inds = self._get_probe_index_for_one_iter_on_off(frames_unique)

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

        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
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

    def generate_movie(self):
        """
        generate movie for display frame by frame
        """

        self.frames = self.generate_frames()

        if self.coordinate == 'degree':
            coord_x = self.monitor.deg_coord_x
            coord_y = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_x = self.monitor.lin_coord_x
            coord_y = self.monitor.lin_coord_y
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
            if curr_frame[0] == 1:  # not a gap

                curr_probes = ([curr_frame[1][0], curr_frame[1][1], curr_frame[2]],)

                if i == 0:  # first frame and (not a gap)
                    curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                      deg_coord_azi=coord_x,
                                                      probes=curr_probes,
                                                      width=self.probe_size[0],
                                                      height=self.probe_size[1],
                                                      ori=self.probe_orientation,
                                                      background_color=self.background)
                else:  # (not first frame) and (not a gap)
                    if self.frames[i - 1][1] is None:  # (not first frame) and (not a gap) and (new square from gap)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          background_color=self.background)
                    elif (curr_frame[1] != self.frames[i - 1][1]).any() or (curr_frame[2] != self.frames[i - 1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          background_color=self.background)

                # assign current display matrix to full sequence
                full_seq[i] = curr_disp_mat

            # add sync square for photodiode
            full_seq[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = curr_frame[3]

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: ' +
                       str(int(100 * (i + 1) / len(self.frames))) + '%')

        # generate log dictionary
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
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    min_distance : float, default 20.
        the minimum distance in visual degree for any pair of probe centers
        in a given frame
    grid_space : 2-tuple of floats, optional
        first coordinate is altitude, second coordinate is azimuth
    probe_size : 2-tuple of floats, optional
        size of flicker probes. First coordinate defines the width, and
        second coordinate defines the height
    probe_orientation : float, optional
        orientation of flicker probes
    probe_frame_num : int, optional
        number of frames for each square presentation
    subregion : list or tuple
        the region on the monitor that will display the sparse noise,
        list or tuple, [min_alt, max_alt, min_azi, max_azi]
    sign : {'ON-OFF', 'ON', 'OFF'}, optional
        determines which pixels appear in the `subregion`, defaults to
        `'ON-Off'` so that both on and off pixels appear. If `'ON` selected
        only on pixels (white) are displayed in the noise `subregion while if
        `'OFF'` is selected only off (black) pixels are displayed in the noise
    iteration : int, optional
        number of times to present stimulus with random order, the total number
        a paticular probe will be displayded will be iteration * repeat,
        defaults to `1`
    repeat : int, optional
        number of repeat of whole sequence, the total number a paticular probe
        will be displayded will be iteration * repeat, defaults to `1`
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    """

    def __init__(self, monitor, indicator, min_distance=20., background=0., coordinate='degree',
                 grid_space=(10., 10.), probe_size=(10., 10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1, repeat=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):
        """
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        super(LocallySparseNoise, self).__init__(monitor=monitor, indicator=indicator,
                                                 background=background, coordinate=coordinate,
                                                 pregap_dur=pregap_dur, postgap_dur=postgap_dur)

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

        if repeat >= 1:
            self.repeat = int(repeat)
        else:
            raise ValueError('repeat should be no less than 1.')

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

        probes_left = list(probes)

        for probe in probes:

            # print len(probes)

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
                probes_left.remove(probe)

        return probes_one_frame, probes_left

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
            curr_frames, all_probes_cpy = self._generate_probe_locs_one_frame(probes=all_probes_cpy)
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
                # print ('redistributing probes among frames: no more probes can be moved.')
                pass
            if probe_diff <= 1:
                # print ('redistributing probes among frames: probes already well distributed.')
                pass

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

            for display_ind in range(display_num):
                index_to_display += [display_ind * 2 + 1] * probe_on_frame_num
                index_to_display += [display_ind * 2 + 2] * probe_off_frame_num

            index_to_display = index_to_display * self.repeat

            index_to_display = [0] * self.pregap_frame_num + index_to_display + [0] * self.postgap_frame_num

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
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
    is_smooth_edge : bool
        True, smooth circle edge with smooth_width_ratio and smooth_func
        False, do not smooth edge
    smooth_width_ratio : float, should be smaller than 1.
        the ratio between smooth band width and radius, circle edge is the middle
        of smooth band
    smooth_func : function object
        this function take two inputs: 1) ndarray storing the distance from each
        pixel to smooth band center; 2) smooth band width.
        returns smoothed mask with same shape as input ndarray
    is_blank_block : bool
        if True, one blank block (full screen background with the same duration of other blocks)
        will be displayed for each iteration. The frames of this condition will be:
        (1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), the meaning of these numbers can be found in
        self.frame_config
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 center=(0., 60.), sf_list=(0.08,), tf_list=(4.,), dire_list=(0.,),
                 con_list=(0.5,), radius_list=(10.,), block_dur=2., midgap_dur=0.5,
                 iteration=1, pregap_dur=2., postgap_dur=3., is_smooth_edge=False,
                 smooth_width_ratio=0.2, smooth_func=blur_cos, is_blank_block=True):
        """
        Initialize `DriftingGratingCircle` stimulus object, inherits Parameters
        from `Stim` class
        """

        super(DriftingGratingCircle, self).__init__(monitor=monitor,
                                                    indicator=indicator,
                                                    background=background,
                                                    coordinate=coordinate,
                                                    pregap_dur=pregap_dur,
                                                    postgap_dur=postgap_dur)

        self.stim_name = 'DriftingGratingCircle'
        if len(center) != 2:
            raise ValueError("DriftingGragingCircle: input 'center' should have "
                             "two elements: (altitude, azimuth).")
        self.center = center
        self.sf_list = list(set(sf_list))
        self.tf_list = list(set(tf_list))
        self.dire_list = list(set(dire_list))
        self.con_list = list(set(con_list))
        self.radius_list = list(set(radius_list))
        self.is_smooth_edge = is_smooth_edge
        self.smooth_width_ratio = smooth_width_ratio
        self.smooth_func = smooth_func

        if int(block_dur * self.monitor.refresh_rate) >= 4:
            self.block_dur = float(block_dur)
        else:
            raise ValueError('There should be more than 4 frames per block, otherwise the '
                             'synchronized indicator strategy will not work.')

        if midgap_dur >= 0.:
            self.midgap_dur = float(midgap_dur)
        else:
            raise ValueError('midgap_dur should be no less than 0 second')

        self.iteration = iteration
        self.frame_config = ('is_display', 'isCycleStart', 'spatial frequency (cycle/deg)',
                             'temporal frequency (Hz)', 'direction (deg)',
                             'contrast [0., 1.]', 'radius (deg)', 'phase (deg)',
                             'indicator color [-1., 1.]')
        self.is_blank_block = bool(is_blank_block)

        for tf in tf_list:
            period = 1. / tf
            if (0.05 * period) < (block_dur % period) < (0.95 * period):
                # print(period)
                # print(block_dur % period)
                # print(0.95 * period)
                error_msg = ('Duration of each block times tf ' + str(tf)
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

        if self.is_blank_block:
            all_conditions.append((0., 0., 0., 0., 0.))

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

        if tf == 0.:
            phases = [0.] * self.block_frame_num
            frame_per_cycle = self.block_frame_num

        else:
            frame_per_cycle = int(self.monitor.refresh_rate / tf)

            phases_per_cycle = list(np.arange(0, np.pi * 2, np.pi * 2 / frame_per_cycle))

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
        off_params = [0, None, None, None, None, None, None, None, -1.]
        # midgap_frames = int(self.midgap_dur*self.monitor.refresh_rate)

        for i in range(self.iteration):
            if i == 0:  # very first block
                frames += [off_params for ind in range(self.pregap_frame_num)]
            else:  # first block for the later iteration
                frames += [off_params for ind in range(self.midgap_frame_num)]

            all_conditions = self._generate_all_conditions()
            random.shuffle(all_conditions)

            for j, condition in enumerate(all_conditions):
                if j != 0:  # later conditions
                    frames += [off_params for ind in range(self.midgap_frame_num)]

                sf, tf, dire, con, size = condition

                # get phase list for each condition
                phases, frame_per_cycle = self._generate_phase_list(tf)
                # if (dire % 360.) >= 90. and (dire % 360. < 270.):
                #      phases = [-phase for phase in phases]

                for k, phase in enumerate(phases):  # each frame in the block

                    # mark first frame of each cycle
                    if k % frame_per_cycle == 0:
                        first_in_cycle = 1
                    else:
                        first_in_cycle = 0

                    frames.append([1, first_in_cycle, sf, tf, dire,
                                   con, size, phase, float(first_in_cycle)])

        # add post gap frame
        frames += [off_params for ind in range(self.postgap_frame_num)]

        # add non-synchronized indicator
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

        if condi_params[0] == 0.: # blank block

            frames_unique_condi = ((1, 1, 0., 0., 0., 0., 0., 1.),
                                   (1, 1, 0., 0., 0., 0., 0., 0.))
            index_to_display_condi = [1] * self.block_frame_num
            index_to_display_condi[0] = 0

        else:

            phases_unique = phases[0:frame_per_cycle]

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
                curr_index_to_display_condi = np.array(condi_dict[condi_key]['index_to_display'])
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

            if frame[0] == 1 and frame[2] != 0.:  # not a gap and not a blank block

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

            # add sync square for photodiode
            mov[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[-1]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        self_dict = dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')
        self_dict.pop('smooth_func')
        log = {'stimulation': self_dict,
               'monitor': mondict,
               'indicator': indicator_dict}

        return mov, log

    def _generate_circle_mask_dict(self):
        """
        generate a dictionary of circle masks for each size in size list
        """

        masks = {}
        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        for radius in self.radius_list:
            curr_mask = get_circle_mask(map_alt=coord_alt, map_azi=coord_azi,
                                        center=self.center, radius=radius,
                                        is_smooth_edge=self.is_smooth_edge,
                                        blur_ratio=self.smooth_width_ratio,
                                        blur_func=self.smooth_func)
            masks.update({radius: curr_mask})

        return masks

    def generate_movie(self):
        """
        Generate movie frame by frame
        """

        self.frames = self.generate_frames()
        mask_dict = self._generate_circle_mask_dict()

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

        mov = np.ones((len(self.frames),
                       coord_azi.shape[0],
                       coord_azi.shape[1]), dtype=np.float32) * self.background
        background_frame = np.ones(coord_azi.shape, dtype=np.float32) * self.background

        for i, curr_frame in enumerate(self.frames):

            if curr_frame[0] == 1 and curr_frame[2] != 0. :  # not a gap and not a blank block

                # curr_ori = self._get_ori(curr_frame[4])
                curr_grating = get_grating(alt_map=coord_alt,
                                           azi_map=coord_azi,
                                           dire=curr_frame[4],
                                           spatial_freq=curr_frame[2],
                                           center=self.center,
                                           phase=curr_frame[7],
                                           contrast=curr_frame[5])
                # plt.imshow(curr_grating)
                # plt.show()

                curr_grating = curr_grating * 2. - 1.  # change scale from [0., 1.] to [-1., 1.]

                curr_circle_mask = mask_dict[curr_frame[6]]

                mov[i] = ((curr_grating * curr_circle_mask) +
                          (background_frame * (curr_circle_mask * -1. + 1.)))

            # add sync square for photodiode
            mov[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = curr_frame[-1]

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: ' +
                       str(int(100 * (i + 1) / len(self.frames))) + '%')

        # generate log dictionary
        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        self_dict = dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')
        self_dict.pop('smooth_func')
        log = {'stimulation': self_dict,
               'monitor': mondict,
               'indicator': indicator_dict}

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
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
    is_smooth_edge : bool
        True, smooth circle edge with smooth_width_ratio and smooth_func
        False, do not smooth edge
    smooth_width_ratio : float, should be smaller than 1.
        the ratio between smooth band width and radius, circle edge is the middle
        of smooth band
    smooth_func : function object
        this function take two inputs: 1) ndarray storing the distance from each
        pixel to smooth band center; 2) smooth band width.
        returns smoothed mask with same shape as input ndarray
    is_blank_block : bool, optional
        if True, a full screen background will be displayed as an additional grating.
        The frames of this condition will be: (1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 or 0.0),
        the meaning of these numbers can be found in self.frame_config
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 center=(0., 60.), sf_list=(0.08,), ori_list=(0., 90.), con_list=(0.5,),
                 radius_list=(10.,), phase_list=(0., 90., 180., 270.), display_dur=0.25,
                 midgap_dur=0., iteration=1, pregap_dur=2., postgap_dur=3.,
                 is_smooth_edge=False, smooth_width_ratio=0.2, smooth_func=blur_cos,
                 is_blank_block=True):
        """
        Initialize `StaticGratingCircle` stimulus object, inherits Parameters
        from `Stim` class
        """

        super(StaticGratingCircle, self).__init__(monitor=monitor,
                                                  indicator=indicator,
                                                  background=background,
                                                  coordinate=coordinate,
                                                  pregap_dur=pregap_dur,
                                                  postgap_dur=postgap_dur)

        self.stim_name = 'StaticGratingCircle'

        if len(center) != 2:
            raise ValueError("StaticGragingCircle: input 'center' should have "
                             "two elements: (altitude, azimuth).")
        self.center = center
        self.sf_list = list(set(sf_list))
        self.phase_list = list(set([p % 360. for p in phase_list]))
        self.ori_list = list(set([o % 180. for o in ori_list]))
        self.con_list = list(set(con_list))
        self.radius_list = list(set(radius_list))
        self.is_smooth_edge = is_smooth_edge
        self.smooth_width_ratio = smooth_width_ratio
        self.smooth_func = smooth_func

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
        self.is_blank_block = bool(is_blank_block)

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
        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        for radius in self.radius_list:
            curr_mask = get_circle_mask(map_alt=coord_alt, map_azi=coord_azi,
                                        center=self.center, radius=radius,
                                        is_smooth_edge=self.is_smooth_edge,
                                        blur_ratio=self.smooth_width_ratio,
                                        blur_func=self.smooth_func)
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

        if self.is_blank_block:
            all_conditions.append((0., 0., 0., 0., 0.))

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

        all_conditions = self._generate_all_conditions()
        gap_frame = (0., None, None, None, None, None, -1.)
        frames_unique = [gap_frame]

        for condition in all_conditions:
            frames_unique.append((1, condition[0], condition[1], condition[2],
                                  condition[3], condition[4], 1.))
            frames_unique.append((1, condition[0], condition[1], condition[2],
                                  condition[3], condition[4], 0.))

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

            if len(frames_unique) % 2 != 1:
                raise ValueError('StaticGratingCircle: the number of unique frames should odd.')
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

            if frame[0] == 1 and frame[1] != 0:  # not a gap and not a blank grating

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
        self_dict.pop('smooth_func')
        log = {'stimulation': self_dict,
               'monitor': mondict,
               'indicator': indicator_dict}

        return mov, log


class StaticImages(Stim):
    """
    Generate static images stimulus

    Stimulus routine presents a sequence of static images in a random order.
    Currently the input image stack should be a tif file. The size of the
    image should be exactly same as the pixel dimension of downsized monitor
    pixel resolution. For example if self.monitor.resolution = (1200,1920)
    and self.monitor.downsample_rate = 10. The shape of input image stack
    should be n x 120 x 192. Value of the input image stack should be within
    the range of [-1., 1.]. The values out of this range will be handled
    by psychopy.visual.ImageStim() function. The reason of this seemingly
    stringent requirement is that, for visual physiological experiments,
    the parameters of visual stimuli should be very well controlled. Any
    imaging cropping, zooming, transformating etc. will affect luminance,
    contrast, spatial resolution etc. and produce unexpected effects.

    This stimulus routing provides a method to generate such image stacks.
    StaticImages.wrap_images() takes a list of image files transform them
    into a desired spherically corrected and luminance normalized image
    stack into visual degree coordinates and save it as a tif file.

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    img_center : 2-tuple of floats, optional
        coordintes for center of the images (altitude, azimuth)
    deg_per_pixel: float, or list/tuple of two floats
        pixel size in visual degrees of unwrapped image (altitude, azimuth),
        if float, assume sizes in altitude and azimuth are the same
    display_dur : float, optional
        duration of each condition in seconds, defaults to `0.25`
    midgap_dur : float, optional
        duration of gap between conditions, defaults to `0.`
    iteration : int, optional
        number of times the stimulus is displayed, defaults to `1`
    is_blank_block : bool, optional
        if True, a full screen background will be displayed as an additional image.
        index of this image will be -1.
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 img_center=(0., 60.), deg_per_pixel=(0.1, 0.1), display_dur=0.25,
                 midgap_dur=0., iteration=1, pregap_dur=2., postgap_dur=3., is_blank_block=True):
        """
        Initialize `StaticImages` stimulus object, inherits Parameters from `Stim` class
        """

        super(StaticImages, self).__init__(monitor=monitor, indicator=indicator,
                                           background=background, coordinate=coordinate,
                                           pregap_dur=pregap_dur, postgap_dur=postgap_dur)

        if len(img_center) != 2:
            raise ValueError("StaticImages: input 'img_center' should have "
                             "two elements: (altitude, azimuth).")
        self.stim_name = 'StaticImages'
        self.img_center = img_center
        self.frame_config = ('is_display', 'image_index', 'indicator color [-1., 1.]')

        try:
            self.deg_per_pixel_alt = float(deg_per_pixel[0])
            self.deg_per_pixel_azi = float(deg_per_pixel[1])
        except TypeError:
            self.deg_per_pixel_alt = self.deg_per_pixel_azi = float(deg_per_pixel)

        self.display_dur = float(display_dur)
        self.midgap_dur = float(midgap_dur)
        self.iteration = int(iteration)
        self.is_blank_block = bool(is_blank_block)

    @property
    def display_frame_num(self):
        return int(self.display_dur * self.monitor.refresh_rate)

    @property
    def midgap_frame_num(self):
        return int(self.midgap_dur * self.monitor.refresh_rate)

    def wrap_images(self, work_dir):
        """
        look for the 'images_original.tif' in the work_dir, load the images,
        warp and luminance correct images, save wrapping results in an HDF5 file
        with name "wrapped_images_for_display.hdf5" in the work_dir

        datasets
        --------
        images_wrapped : 3d array, frame x altitude x azimuth,
            each frame will have  same shape as the pixel resolution of down
            sampled self.monitor

            attrs
            +++++
            altitude : 2d array, altitude x azimuth
                altitude coordinates of wrapped images in visual degrees,
                same shape as each frame of images_wrapped
            azimuth : 2d array, altitude x azimuth
                azimuth coordinates of wrapped images in visual degrees,
                same shape as each frame of images_wrapped

        images_dewrapped : 3d array, frame x altitude x azimuth
            dewrapped images, please note there is no pixel to pixel relationship
            between images_wrapped and images_dewrapped. Different regions in
            images_dewrapped have different sampling density to generate
            images_wrapped. Some pixels in image_dewrapped (especially on the edge)
            may not get presented by image_wrapped. images_dewrapped represent the
            minimum rectangle region in the original image that cover the entire
            images_wrapped.

            attrs
            +++++
            altitude : 2d array, altitude x azimuth
                altitude coordinates of dewrapped images in visual degrees,
                same shape as each frame in images_dewrapped
            azimuth : 2d array, altitude x azimuth
                azimuth coordinates of dewrapped images in visual degrees,
                same shape as each frame in images_dewrapped
        """

        if os.path.isfile(os.path.join(work_dir, 'wrapped_images_for_display.hdf5')):
            raise IOError('"wrapped_images_for_display.hdf5" already exists in the '
                          '"work_dir" : {}. Please choose another folder or delete '
                          'the file.'.format(os.path.realpath(work_dir)))

        imgs = tf.imread(os.path.join(work_dir, 'images_original.tif'))

        deg_per_pixel = [self.deg_per_pixel_alt, self.deg_per_pixel_azi]
        wrapping_results = self.monitor.warp_images(imgs=imgs, center_coor=self.img_center,
                                                    deg_per_pixel=deg_per_pixel,
                                                    is_luminance_correction=True)
        imgs_w, alt_w, azi_w, imgs_dw, alt_dw, azi_dw = wrapping_results
        results_f = h5py.File(os.path.join(work_dir, 'wrapped_images_for_display.hdf5'))
        grp_w = results_f.create_group('images_wrapped')
        _ = grp_w.create_dataset('images', data=imgs_w)
        _ = grp_w.create_dataset('altitude', data=alt_w)
        _ = grp_w.create_dataset('azimuth', data=azi_w)
        grp_dw = results_f.create_group('images_dewrapped')
        _ = grp_dw.create_dataset('images', data=imgs_dw)
        _ = grp_dw.create_dataset('altitude', data=alt_dw.astype(np.float32))
        _ = grp_dw.create_dataset('azimuth', data=azi_dw.astype(np.float32))
        results_f.close()

    def set_imgs_from_tif(self, imgs_path_wrapped, imgs_path_dewrapped=None):

        imgs_wrapped = tf.imread(imgs_path_wrapped)

        if len(imgs_wrapped.shape) != 3:
            raise ValueError('StaticImages: the input wrapped images should be a 3d array.')

        if (imgs_wrapped.shape[1], imgs_wrapped.shape[2]) != self.monitor.deg_coord_x.shape:
            raise ValueError('StaticImages: the input wrapped images should have '
                             'the same dimensions of the pixel resolution of '
                             'downsampled monitor.')

        self.images_wrapped = imgs_wrapped

        if imgs_path_dewrapped is not None:

            imgs_dewrapped = tf.imread(imgs_path_dewrapped)

            if imgs_dewrapped.shape[0] != imgs_wrapped.shape[0]:
                print ('The input dewrapped images have different dimensions from the '
                       'input wrapped images. Set self.images_dewrapped to None.')
                self.images_dewrapped = None
            else:
                self.images_dewrapped = tf.imread(imgs_path_dewrapped)
        else:
            self.images_dewrapped = None

    def set_imgs_from_hdf5(self, imgs_file_path):
        """
        set 3d arrays from a hdf5 file for display. Ideally the hdf5 file should be
        the result from self.wrap_images() method. Only designed to work with wrapped
        images

        parameters
        ----------
        imgs_file_path : str
            system path ot the hdf5 file. It should have at least one dataset named
            'images_wrapped' containing a 3d array of wrapped images to display
        """
        img_f = h5py.File(imgs_file_path, 'r')

        if len(img_f['images_wrapped/images'].shape) != 3:
            raise ValueError('StaticImages: the input wrapped images should be a 3d array.')

        if (img_f['images_wrapped/images'].shape[1],
            img_f['images_wrapped/images'].shape[2]) != self.monitor.deg_coord_x.shape:
            raise ValueError('StaticImages: the input wrapped images should have '
                             'the same dimensions of the pixel resolution of '
                             'downsampled monitor.')

        try:
            alt_w = img_f['images_wrapped/altitude'].value
        except:
            alt_w = None

        try:
            azi_w = img_f['images_wrapped/azimuth'].value
        except:
            azi_w = None

        if alt_w is not None:
            if not np.array_equal(alt_w, self.monitor.deg_coord_y):
                raise ValueError('the altitude coordinates of input wrapped images do not '
                                 'match the wrapped monitor pixel altitude coordinates.')
        if azi_w is not None:
            if not np.array_equal(azi_w, self.monitor.deg_coord_x):
                raise ValueError('the azimuth coordinates of input wrapped images do not '
                                 'match the wrapped monitor pixel azimuth coordinates.')

        self.images_wrapped = img_f['images_wrapped/images'].value

        if 'images_dewrapped' in img_f:
            if not img_f['images_dewrapped/images'].shape != 3:
                print ('The images_dewrapped in the input file is not 3d. '
                       'Set self.images_dewrapped to None.')
                self.images_dewrapped = None
                self.altitude_dewrapped = None
                self.azimuth_dewrapped = None

            elif img_f['images_dewrapped/images'].shape[0] != self.images_wrapped.shape[0]:
                print ('The number of frames of images_dewrapped in the input file is different'
                       'from the number of frames of self.images. Set self.images_dewrapped to None.')
                self.images_dewrapped = None
                self.altitude_dewrapped = None
                self.azimuth_dewrapped = None
            else:
                self.images_dewrapped = img_f['images_dewrapped/images'].value
                try:
                    alt_d = img_f['images_dewrapped/altitude'].value
                    if alt_d.shape[0] != self.images_dewrapped.shape[1] or \
                                    alt_d.shape[1] != self.images_dewrapped.shape[2]:
                        print ('altitude coordinates of images_dewrapped in the input file have '
                               'different shape as frames in self.images_dewrapped. Set'
                               'self.altitude_dewrapped to None.')
                        self.altitude_dewrapped = None
                    else:
                        self.altitude_dewrapped = alt_d
                except:
                    self.altitude_dewrapped = None

                try:
                    azi_d = img_f['images_dewrapped/azimuth'].value
                    if azi_d.shape[0] != self.images_dewrapped.shape[1] or \
                                    azi_d.shape[1] != self.images_dewrapped.shape[2]:
                        print ('azimuth coordinates of images_dewrapped in the input file have '
                               'different shape as frames in self.images_dewrapped. Set'
                               'self.azimuth_dewrapped to None.')
                        self.azimuth_dewrapped = None
                    else:
                        self.azimuth_dewrapped = azi_d
                except:
                    self.azimuth_dewrapped = None

        else:
            print ('Cannot find "images_dewrapped" dataset in the input file. '
                   'Set self.images_dewrapped to None.')
            self.images_dewrapped = None
            self.altitude_dewrapped = None
            self.azimuth_dewrapped = None

        img_f.close()

    def _generate_frames_for_index_display(self):
        """
        generate a tuple of unique frames, each element of the tuple
        represents a unique display condition including gap

        frame structure:
            0. is_display: if gap --> 0; if display --> 1
            1. image index, non-negative integer
            2. indicator color, [-1., 1.]
        """
        if not hasattr(self, 'images_wrapped'):
            raise LookupError('StaticImages: cannot find attribute: "imgs_wrapped".'
                              'Please use self.set_imgs_from_tif() or '
                              'self.set_imgs_from_hdf5() to set the images.')

        img_num = self.images_wrapped.shape[0]
        frames_unique = [(0, None, -1.)]

        for i in range(img_num):
            frames_unique.append((1, i, 1.))
            frames_unique.append((1, i, 0.))

        # adding blank image
        if self.is_blank_block:
            frames_unique.append((1, -1, 1.))
            frames_unique.append((1, -1, 0.))

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

            if len(frames_unique) % 2 != 1:
                raise ValueError('StaticGratingCircle: the number of unique frames should odd.')
            img_num = (len(frames_unique) - 1) / 2

            index_to_display = [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                display_sequence = range(img_num)
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

        mov = self.background * np.ones((len(self.frames_unique),
                                         self.images_wrapped.shape[1],
                                         self.images_wrapped.shape[2]),
                                        dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):

            if frame[0] == 1 and frame[1] != -1:  # not a gap and not a blank block

                curr_img = self.images_wrapped[frame[1]]
                curr_img[np.isnan(curr_img)] = self.background

                mov[i] = curr_img

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


class StimulusSeparator(Stim):
    """
    a quick flash of indicator to separate different
    visual stimuli when displayed in the same session

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    indicator_on_frame_num : int
        number of frames the indicator is white, should be positive.
    indicator_off_frame_num : int
        number of frames the indicator is black, should be positive.
    cycle_num : int
        number of repeat of the indicator flash, should be positive.

    """

    def __init__(self, monitor, indicator, coordinate='degree', background=0.,
                 indicator_on_frame_num=4, indicator_off_frame_num=4,
                 cycle_num=10, pregap_dur=0., postgap_dur=0.):
        """
        Initialize `StimulusSeparator` stimulus object, inherits Parameters from `Stim` class
        """

        super(StimulusSeparator, self).__init__(monitor=monitor,
                                                indicator=indicator,
                                                background=background,
                                                coordinate=coordinate,
                                                pregap_dur=pregap_dur,
                                                postgap_dur=postgap_dur)

        self.stim_name = 'StimulusSeparator'
        self.background = float(background)
        self.indicator_on_frame_num = int(indicator_on_frame_num)
        self.indicator_off_frame_num = int(indicator_off_frame_num)
        self.cycle_num = int(cycle_num)
        self.frame_config = ('is_display', 'indicator color [-1., 1.]')

    def _generate_frames_for_index_display(self):
        """
        frame structure is as following

        first element: is_display
        second element: indicator color
        """
        return ((0, -1), (1, 1.), (1, -1.))

    def _generate_display_index(self):

        if self.indicator.is_sync:
            frames_unique = self._generate_frames_for_index_display()
            index_to_display = [0] * self.pregap_frame_num

            for cycle_ind in range(self.cycle_num):
                index_to_display += [1] * self.indicator_on_frame_num
                index_to_display += [2] * self.indicator_off_frame_num

            index_to_display += [0] * self.postgap_frame_num
            return frames_unique, index_to_display
        else:
            raise NotImplementedError, "method not available for non-sync indicator."

    def generate_movie_by_index(self):

        self.frames_unique, self.index_to_display = self._generate_display_index()

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

        mov = self.background * np.ones((len(self.frames_unique),
                                         coord_azi.shape[0],
                                         coord_azi.shape[1]),
                                        dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
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


class CombinedStimuli(Stim):
    """
    the stimulus class that can combine different stimuli into one session.

    example:
    >>> import retinotopic_mapping.StimulusRoutines as stim
    >>> from retinotopic_mapping.MonitorSetup import Monitor, Indicator
    >>> from retinotopic_mapping.DisplayStimulus import DisplaySequence
    >>> mon = Monitor(resolution=(1200, 1920), dis=15., mon_width_cm=52., mon_height_cm=32.)
    >>> ind = Indicator(mon)
    >>> uc = stim.UniformContrast(mon, ind, duration=10., color=-1.)
    >>> ss = stim.StimulusSeparator(mon, ind)
    >>> cs = stim.CombinedStimuli(mon, ind)
    >>> cs.set_stimuli([ss, uc, ss])
    >>> ds = DisplaySequence(log_dir='C:/data')
    >>> ds.set_stim(cs)
    >>> ds.trigger_display()

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
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    """
    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 pregap_dur=2., postgap_dur=3.):

        super(CombinedStimuli, self).__init__(monitor=monitor, indicator=indicator,
                                              background=background, coordinate=coordinate,
                                              pregap_dur=pregap_dur, postgap_dur=postgap_dur)

        self.stim_name = 'CombinedStimuli'

    def set_stimuli(self, stimuli, static_images_path=None):
        """

        parameters
        ----------
        stimuli : list of above stimulus object
        static_images_path : str
            system path to the hdf5 file storing the wrapped images for display. If there
            is StaticImages stimulus in the stimuli list, it will try to load images and
            display
        """

        for stimulus in stimuli:
            if not stimulus.stim_name in ['UniformContrast', 'FlashingCircle', 'SparseNoise',
                                          'LocallySparseNoise', 'DriftingGratingCircle',
                                          'StaticGratingCircle', 'StaticImages', 'StimulusSeparator']:
                raise LookupError('Stimulus type "{}" is not currently supported.'
                                  .format(stimulus.stim_name))

        self.stimuli = stimuli
        self.static_images_path = static_images_path

    def generate_movie_by_index(self):

        t0 = time.time()
        print ('\n{:04.1f} min : CombinedStimulus: generating stimuli ...'.format(time.time() - t0))

        self.frames_unique = []
        self.index_to_display = []
        self.individual_logs = {}
        mov = []

        curr_start_frame_ind = 0

        for stim_ind, stimulus in enumerate(self.stimuli):

            curr_stim_name = stimulus.stim_name
            curr_stim_id = ft.int2str(stim_ind, 3) + '_' + curr_stim_name

            stimulus.set_monitor(self.monitor)
            stimulus.set_indicator(self.indicator)
            stimulus.set_pregap_dur(self.pregap_dur)
            stimulus.set_postgap_dur(self.postgap_dur)
            stimulus.set_background(self.background)
            stimulus.set_coordinate(self.coordinate)

            # load the images if the stimulus is StaticImages
            if curr_stim_name == 'StaticImages':
                stimulus.set_imgs_from_hdf5(imgs_file_path=self.static_images_path)

            curr_mov, curr_log = stimulus.generate_movie_by_index()
            curr_log.pop('monitor')
            curr_log.pop('indicator')

            self.individual_logs.update({curr_stim_id: curr_log['stimulation']})

            curr_frames_unique = [[curr_stim_id] + list(f) for f in curr_log['stimulation']['frames_unique']]
            curr_index_to_display = np.array(curr_log['stimulation']['index_to_display'], dtype=np.uint64)

            self.frames_unique += curr_frames_unique
            self.index_to_display.append(curr_index_to_display + curr_start_frame_ind)
            mov.append(curr_mov)

            curr_start_frame_ind += len(curr_frames_unique)

            print ('{:04.1f} min : stimulus: {:<30}; estimated display duration: {:4.1f} minute(s).'
                   .format((time.time() - t0) / 60., curr_stim_id,
                           len(curr_index_to_display) / (60. * self.monitor.refresh_rate)))

        self.frames_unique = tuple([tuple(f) for f in self.frames_unique])
        self.index_to_display = list(np.concatenate(self.index_to_display, axis=0))
        mov = np.concatenate(mov, axis=0)

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        self_dict = dict(self.__dict__)
        self_dict.pop('monitor')
        self_dict.pop('indicator')

        stim_seq = []
        for stim_ind, stim in enumerate(self.stimuli):
            stim_seq.append(ft.int2str(stim_ind, 3) + '_' + stim.stim_name)
        self_dict.pop('stimuli')
        self_dict.update({'stimuli_sequence':stim_seq})
        log = {'stimulation': self_dict,
               'monitor': mondict,
               'indicator': indicator_dict}

        return mov, log

    def clear(self):
        super(CombinedStimuli, self).clear()
        if hasattr(self, 'stimuli'):
            del self.stimuli
        if hasattr(self, 'static_images_path'):
            del self.static_images_path


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
                 square_size=25., square_center=(0, 0), flicker_frame=10,
                 sweep_width=20., step_width=0.15, direction='B2U', sweep_frame=1,
                 iteration=1, pregap_dur=2., postgap_dur=3.):

        super(KSstim, self).__init__(monitor=monitor,
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

        neg_x = np.ceil(abs(((min_x - self.square_center[0]) /
                             (2 * self.square_size)))) + 1
        pos_x = np.ceil(abs(((max_x - self.square_center[0]) /
                             (2 * self.square_size)))) + 1

        neg_y = np.ceil(abs(((min_y - self.square_center[0]) /
                             (2 * self.square_size)))) + 1
        pos_y = np.ceil(abs(((max_y - self.square_center[0]) /
                             (2 * self.square_size)))) + 1

        squareV = np.ones((np.size(map_x, 0),
                           np.size(map_x, 1)),
                          dtype=np.float32)
        squareV = -1 * squareV

        stepV = np.arange(self.square_center[0] - (2 * neg_x + 0.5) * self.square_size,
                          self.square_center[0] + (2 * pos_x - 0.5) * self.square_size,
                          self.square_size * 2)

        for i in range(len(stepV)):
            squareV[np.where(np.logical_and(map_x >= stepV[i],
                                            map_x < (stepV[i] +
                                                     self.square_size)))] = 1.0

        squareH = np.ones((np.size(map_y, 0),
                           np.size(map_y, 1)), dtype=np.float32)
        squareH = -1 * squareH

        stepH = np.arange(self.square_center[1] - (2 * neg_y + 0.5) * self.square_size,
                          self.square_center[1] + (2 * pos_y - 0.5) * self.square_size,
                          self.square_size * 2)

        for j in range(len(stepH)):
            squareH[np.where(np.logical_and(map_y >= stepH[j],
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
        step_width = self.step_width
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
                               np.size(map_x, 1)), dtype=np.float32)
            for i in range(len(step_x)):
                temp = sweeps[i, :, :]
                temp[np.where(np.logical_and(map_x >= step_x[i],
                                             map_x < (step_x[i] +
                                                      sweep_width)))] = 1.0
                sweep_table.append(('V', step_x[i], step_x[i] + sweep_width))
                del temp

        if 'step_y' in locals():
            sweeps = np.zeros((len(step_y),
                               np.size(map_y, 0),
                               np.size(map_y, 1)), dtype=np.float32)
            for j in range(len(step_y)):
                temp = sweeps[j, :, :]
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

        sweep_num = np.size(sweeps, 0)  # Number of sweeps vertical or horizontal
        displayframe_num = sweep_frame * sweep_num  # total frame number for 1 iter

        # frames for one iteration
        iter_frames = []

        # add frames for gaps
        for i in range(self.pregap_frame_num):
            iter_frames.append([0, None, None, -1])

        # add frames for display
        is_reverse = []

        for i in range(displayframe_num):

            if (np.floor(i // flicker_frame)) % 2 == 0:
                is_reverse = -1
            else:
                is_reverse = 1

            sweep_index = int(np.floor(i // sweep_frame))

            # add sychornized indicator
            if self.indicator.is_sync == True:
                indicator_color = 1
            else:
                indicator_color = -1

            iter_frames.append([1, is_reverse, sweep_index, indicator_color])

        # add gap frames at the end
        for i in range(self.postgap_frame_num):
            iter_frames.append([0, None, None, -1])

        full_frames = []

        # add frames for multiple iteration
        for i in range(int(iteration)):
            full_frames += iter_frames

        # add non-synchronized indicator
        if self.indicator.is_sync == False:
            indicator_frame = self.indicator.frame_num

            for j in range(np.size(full_frames, 0)):
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

        self.frames = self.generate_frames()

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
                              np.size(self.monitor.deg_coord_x, 1)),
                             dtype=np.float32) * self.background

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

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: ' + str(int(100 * (i + 1) / len(self.frames))) + '%')

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        KSdict = dict(self.__dict__)
        KSdict.pop('monitor')
        KSdict.pop('indicator')
        full_dict = {'stimulation': KSdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict

    def clear(self):
        self.sweep_table = None
        self.frames = None
        self.square = None

    def set_direction(self, direction):

        if direction in ['B2U', 'U2B', 'L2R', 'R2L']:
            self.direction = direction
            self.clear()
        else:
            raise LookupError, '`direction` not in {"B2U","U2B","L2R","R2L"}'

    def set_sweep_sigma(self, sweepSigma):
        self.sweepSigma = sweepSigma
        self.clear()

    def set_sweep_width(self, sweep_width):
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
                 square_size=25, square_center=(0, 0), flicker_frame=6, sweep_width=20.,
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
        KS_stim = KSstim(self.monitor,
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

        mov = np.vstack((mov_B2U, mov_U2B, mov_L2R, mov_R2L))
        log = {'monitor': dict_B2U['monitor'],
               'indicator': dict_B2U['indicator']}
        stimulation = dict(dict_B2U['stimulation'])
        stimulation['stim_name'] = 'KSstimAllDir'
        stimulation['direction'] = ['B2U', 'U2B', 'L2R', 'R2L']

        sweep_table = []
        frames = []

        sweep_table_B2U = dict_B2U['stimulation']['sweep_table']
        frames_B2U = dict_B2U['stimulation']['frames']
        sweep_length_B2U = len(sweep_table_B2U)
        sweep_table_B2U = [['B2U', x[1], x[2]] for x in sweep_table_B2U]
        frames_B2U = [[x[0], x[1], x[2], x[3], 'B2U'] for x in frames_B2U]
        sweep_table += sweep_table_B2U
        frames += frames_B2U

        sweep_table_U2B = dict_U2B['stimulation']['sweep_table']
        frames_U2B = dict_U2B['stimulation']['frames']
        sweep_length_U2B = len(sweep_table_U2B)
        sweep_table_U2B = [['U2B', x[1], x[2]] for x in sweep_table_U2B]
        frames_U2B = [[x[0], x[1], x[2], x[3], 'U2B'] for x in frames_U2B]
        for frame in frames_U2B:
            if frame[2] is not None:
                frame[2] += sweep_length_B2U
        sweep_table += sweep_table_U2B
        frames += frames_U2B

        sweep_table_L2R = dict_L2R['stimulation']['sweep_table']
        frames_L2R = dict_L2R['stimulation']['frames']
        sweep_length_L2R = len(sweep_table_L2R)
        sweep_table_L2R = [['L2R', x[1], x[2]] for x in sweep_table_L2R]
        frames_L2R = [[x[0], x[1], x[2], x[3], 'L2R'] for x in frames_L2R]
        for frame in frames_L2R:
            if frame[2] is not None:
                frame[2] += sweep_length_B2U + sweep_length_U2B
        sweep_table += sweep_table_L2R
        frames += frames_L2R

        sweep_table_R2L = dict_R2L['stimulation']['sweep_table']
        frames_R2L = dict_R2L['stimulation']['frames']
        sweep_table_R2L = [['R2L', x[1], x[2]] for x in sweep_table_R2L]
        frames_R2L = [[x[0], x[1], x[2], x[3], 'R2L'] for x in frames_R2L]
        for frame in frames_R2L:
            if frame[2] is not None:
                frame[2] += sweep_length_B2U + sweep_length_U2B + sweep_length_L2R
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
