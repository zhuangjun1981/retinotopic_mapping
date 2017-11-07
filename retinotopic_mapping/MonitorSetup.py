# -*- coding: utf-8 -*-
"""
Notes
-----

"""
import numpy as np
import matplotlib.pyplot as plt
import retinotopic_mapping.tools.ImageAnalysis as ia


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
    center_coordinates : tuple of two floats
        (altitude, azimuth), in degrees. the coordinates of the projecting point
        from the eye ball to the monitor. This allows to place the display monitor
        in any arbitrary position.
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

    def __init__(self,
                 resolution,
                 dis,
                 mon_width_cm,
                 mon_height_cm,
                 C2T_cm,
                 C2A_cm,
                 center_coordinates=(0., 60.),
                 visual_field='right',
                 deg_coord_x=None,
                 deg_coord_y=None,
                 name='testMonitor',
                 gamma=None,
                 gamma_grid=None,
                 luminance=None,
                 downsample_rate=10,
                 refresh_rate=60.):
        """
        Initialize monitor object.

        """

        if resolution[0] % downsample_rate != 0 \
                or resolution[1] % downsample_rate != 0:
            raise ArithmeticError('Resolution pixel numbers are not '
                                  'divisible by down sampling rate.')

        self.resolution = resolution
        self.dis = dis
        self.mon_width_cm = mon_width_cm
        self.mon_height_cm = mon_height_cm
        self.C2T_cm = C2T_cm
        self.C2A_cm = C2A_cm
        self.center_coordinates = center_coordinates
        self.visual_field = visual_field
        self.deg_coord_x = deg_coord_x
        self.deg_coord_y = deg_coord_y
        self.name = name
        self.downsample_rate = downsample_rate
        self.gamma = gamma
        self.gamma_grid = gamma_grid
        self.luminance = luminance
        self.refresh_rate = 60

        # distance form projection point of the eye to bottom of the monitor
        self.C2B_cm = self.mon_height_cm - self.C2T_cm
        # distance form projection point of the eye to right of the monitor
        self.C2P_cm = self.mon_width_cm - self.C2A_cm

        resolution = [0, 0]
        resolution[0] = self.resolution[0] / downsample_rate
        resolution[1] = self.resolution[1] / downsample_rate

        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]),
                                               range(resolution[0]))

        if self.visual_field == "left":
            map_x = np.linspace(self.C2A_cm, -1.0 * self.C2P_cm, resolution[1])

        if self.visual_field == "right":
            map_x = np.linspace(-1 * self.C2A_cm, self.C2P_cm, resolution[1])

        map_y = np.linspace(self.C2T_cm, -1.0 * self.C2B_cm, resolution[0])
        old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse=False)

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
            raise ArithmeticError('Resolution pixel numbers are not divisible by down sampling rate.')

        self.downsample_rate = downsample_rate

        resolution = [0, 0]
        resolution[0] = self.resolution[0] / downsample_rate
        resolution[1] = self.resolution[1] / downsample_rate

        # map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]),
        #                                        range(resolution[0]))

        if self.visual_field == "left":
            map_x = np.linspace(self.C2A_cm, -1.0 * self.C2P_cm, resolution[1])

        if self.visual_field == "right":
            map_x = np.linspace(-1 * self.C2P_cm, self.C2P_cm, resolution[1])

        map_y = np.linspace(self.C2T_cm, -1.0 * self.C2B_cm, resolution[0])
        old_map_x, old_map_y = np.meshgrid(map_x, map_y, sparse=False)

        self.lin_coord_x = old_map_x
        self.lin_coord_y = old_map_y

        self.remap()

    def remap(self):
        """
        warp the linear pixel coordinates to a spherical corrected representation.

        Function is called when the monitor object is initialized and populate
        the `deg_coord_x` and `deg_coord_y` attributes.
        """

        resolution = [0, 0]
        resolution[0] = self.resolution[0] / self.downsample_rate
        resolution[1] = self.resolution[1] / self.downsample_rate

        map_coord_x, map_coord_y = np.meshgrid(range(resolution[1]),
                                               range(resolution[0]))

        new_map_x = np.zeros(resolution, dtype=np.float32)
        new_map_y = np.zeros(resolution, dtype=np.float32)

        for j in range(resolution[1]):
            new_map_x[:, j] = ((180.0 / np.pi) *
                               np.arctan(self.lin_coord_x[0, j] / self.dis))
            dis2 = np.sqrt(np.square(self.dis) +
                           np.square(self.lin_coord_x[0, j]))

            for i in range(resolution[0]):
                new_map_y[i, j] = ((180.0 / np.pi) *
                                   np.arctan(self.lin_coord_y[i, 0] / dis2))

        self.deg_coord_x = new_map_x + self.center_coordinates[1]
        self.deg_coord_y = new_map_y + self.center_coordinates[0]

    def plot_map(self):

        resolution = [0, 0]
        resolution[0] = self.resolution[0] / self.downsample_rate
        resolution[1] = self.resolution[1] / self.downsample_rate

        mapcorX, mapcorY = np.meshgrid(range(resolution[1]), range(resolution[0]))

        f1 = plt.figure(figsize=(12, 7))
        f1.suptitle('Remap monitor', fontsize=14, fontweight='bold')

        OMX = plt.subplot(221)
        OMX.set_title('Linear Map X (cm)')
        currfig = plt.imshow(self.lin_coord_x)
        levels1 = range(int(np.floor(self.lin_coord_x.min() / 10) * 10),
                        int((np.ceil(self.lin_coord_x.max() / 10) + 1) * 10), 10)
        im1 = plt.contour(mapcorX, mapcorY, self.lin_coord_x, levels1, colors='k', linewidth=2)
        #        plt.clabel(im1, levels1, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig, ticks=levels1)
        plt.gca().set_axis_off()

        OMY = plt.subplot(222)
        OMY.set_title('Linear Map Y (cm)')
        currfig = plt.imshow(self.lin_coord_y)
        levels2 = range(int(np.floor(self.lin_coord_y.min() / 10) * 10),
                        int((np.ceil(self.lin_coord_y.max() / 10) + 1) * 10), 10)
        im2 = plt.contour(mapcorX, mapcorY, self.lin_coord_y, levels2, colors='k', linewidth=2)
        #        plt.clabel(im2, levels2, fontsize = 10, inline = 1, fmt='%2.2f')
        f1.colorbar(currfig, ticks=levels2)
        plt.gca().set_axis_off()

        NMX = plt.subplot(223)
        NMX.set_title('Spherical Map X (deg)')
        currfig = plt.imshow(self.deg_coord_x)
        levels3 = range(int(np.floor(self.deg_coord_x.min() / 10) * 10),
                        int((np.ceil(self.deg_coord_x.max() / 10) + 1) * 10), 10)
        im3 = plt.contour(mapcorX, mapcorY, self.deg_coord_x, levels3, colors='k', linewidth=2)
        #        plt.clabel(im3, levels3, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig, ticks=levels3)
        plt.gca().set_axis_off()
        #
        NMY = plt.subplot(224)
        NMY.set_title('Spherical Map Y (deg)')
        currfig = plt.imshow(self.deg_coord_y)
        levels4 = range(int(np.floor(self.deg_coord_y.min() / 10) * 10),
                        int((np.ceil(self.deg_coord_y.max() / 10) + 1) * 10), 10)
        im4 = plt.contour(mapcorX, mapcorY, self.deg_coord_y, levels4, colors='k', linewidth=2)
        #        plt.clabel(im4, levels4, fontsize = 10, inline = 1, fmt='%2.1f')
        f1.colorbar(currfig, ticks=levels4)
        plt.gca().set_axis_off()

    def generate_lookup_table(self):
        """
        generate lookup talbe between degree corrdinates and linear corrdinates
        return two matrix:
        lookupI: i index in linear matrix to this pixel after warping
        lookupJ: j index in linear matrix to this pixel after warping
        """

        # length of one degree on monitor at gaze point
        degDis = np.tan(np.pi / 180) * self.dis

        # generate degree coordinate without warpping
        degNoWarpCorX = self.lin_coord_x / degDis
        degNoWarpCorY = self.lin_coord_y / degDis

        # deg coordinates
        degCorX = self.deg_coord_x + self.center_coordinates[0]
        degCorY = self.deg_coord_y + self.center_coordinates[1]

        lookupI = np.zeros(degCorX.shape).astype(np.int32)
        lookupJ = np.zeros(degCorX.shape).astype(np.int32)

        for j in xrange(lookupI.shape[1]):
            currDegX = degCorX[0, j]
            diffDegX = degNoWarpCorX[0, :] - currDegX
            IndJ = np.argmin(np.abs(diffDegX))
            lookupJ[:, j] = IndJ

            for i in xrange(lookupI.shape[0]):
                currDegY = degCorY[i, j]
                diffDegY = degNoWarpCorY[:, IndJ] - currDegY
                indI = np.argmin(np.abs(diffDegY))
                lookupI[i, j] = indI

        return lookupI, lookupJ

    def warp_images(self, imgs, center_coor, deg_per_pixel=0.1, is_luminance_correction=True):
        """
        warp a image stack into visual degree coordinate system

        parameters
        ----------
        imgs : ndarray
            should be 2d or 3d, if 3d, axis will be considered as frame x rows x width
        center_coor : list or tuple of two floats
            the visual degree coordinates of the center of the image (altitude, azimuth)
        deg_per_pixel : float or list/tuple of two floats
            size of original pixel in visual degrees, (altitude, azimuth), if float, assume
            sizes in both dimension are the same
        is_luminance_correction : bool
            if True, wrapped images will have mean intensity equal 0, and values will be
            scaled up to reach minimum equal -1. or maximum equal 1.

        returns
        -------
        imgs_wrapped : 3d array, np.float32
            wrapped images, each frame should have exact same size of down sampled monitor
            resolution. the region on the monitor not covered by the image will have value
            of np.nan. value range [-1., 1.]
        coord_alt_wrapped : 2d array, np.float32
            the altitude coordinates of all pixels in the wrapped images in visual degrees.
            should have the same shape as each frame in 'imgs_wrapped'.
        coord_azi_wrapped : 2d array, np.float32
            the azimuth coordinates of all pixels in the wrapped images in visual degrees.
            should have the same shape as each frame in 'imgs_wrapped'.
        imgs_dewrapped : 3d array, dtype same as imgs
            unwrapped images, same dimension as input image stack. the region of original
            image that was not got displayed (outside of the monitor) will have value of
            np.nan. value range [-1., 1.]
        coord_alt_dewrapped : 2d array, np.float32
            the altitude coordinates of all pixels in the dewrapped images in visual degrees.
            should have the same shape as each frame in 'imgs_dewrapped'.
        coord_azi_dewrapped : 2d array, np.float32
            the azimuth coordinates of all pixels in the dewrapped images in visual degrees.
            should have the same shape as each frame in 'imgs_dewrapped'.
        """

        try:
            deg_per_pixel_alt = abs(float(deg_per_pixel[0]))
            deg_per_pixel_azi = abs(float(deg_per_pixel[1]))
        except TypeError:
            deg_per_pixel_alt = deg_per_pixel_azi = deg_per_pixel

        if len(imgs.shape) == 2:
            imgs_raw = np.array([imgs])
        elif len(imgs.shape) == 3:
            imgs_raw = imgs
        else:
            raise ValueError('input "imgs" should be 2d or 3d array.')

        # generate raw image pixel coordinates in visual degrees
        alt_start = center_coor[0] + (imgs_raw.shape[1] / 2) * deg_per_pixel_alt
        alt_axis = alt_start - np.arange(imgs_raw.shape[1]) * deg_per_pixel_alt
        azi_start = center_coor[1] - (imgs_raw.shape[2] / 2) * deg_per_pixel_azi
        azi_axis = np.arange(imgs_raw.shape[2]) * deg_per_pixel_azi + azi_start
        # img_coord_azi, img_coord_alt = np.meshgrid(azi_axis, alt_axis)

        # initialize output array
        imgs_wrapped = np.zeros((imgs_raw.shape[0],
                                 self.deg_coord_x.shape[0],
                                 self.deg_coord_x.shape[1]), dtype=np.float32)
        imgs_wrapped[:] = np.nan

        # for cropping imgs_raw
        x_min = None;
        x_max = None;
        y_min = None;
        y_max = None

        # for testing
        # img_count = np.zeros((imgs_raw.shape[1], imgs_raw.shape[2]), dtype=np.uint32)

        # loop through every display (wrapped) pixel
        for ii in range(self.deg_coord_x.shape[0]):
            for jj in range(self.deg_coord_x.shape[1]):

                # the wrapped coordinate of current display pixel [alt, azi]
                coord_w = [self.deg_coord_y[ii, jj], self.deg_coord_x[ii, jj]]

                # if the wrapped coordinates of current display pixel is covered
                # by the raw image
                if alt_axis[0] >= coord_w[0] >= alt_axis[-1] and \
                                        azi_axis[0] <= coord_w[1] <= azi_axis[-1]:

                    # get raw pixels arround the wrapped coordinates of current display pixel
                    u = (alt_axis[0] - coord_w[0]) / deg_per_pixel_alt
                    l = (coord_w[1] - azi_axis[0]) / deg_per_pixel_azi

                    # for testing:
                    # img_count[int(u), int(l)] += 1

                    if (u == round(u) and l == round(l)):  # right hit on one raw pixel
                        imgs_wrapped[:, ii, jj] = imgs_raw[:, int(u), int(l)]

                        # for cropping
                        if x_min is None:
                            x_min = x_max = l
                            y_min = y_max = u
                        else:
                            x_min = min(x_min, l)
                            x_max = max(x_max, l)
                            y_min = min(y_min, u)
                            y_max = max(y_max, u)

                    else:
                        u = int(u);
                        b = u + 1;
                        l = int(l);
                        r = l + 1
                        w_ul = 1. / ia.distance(coord_w, [alt_axis[u], azi_axis[l]])
                        w_bl = 1. / ia.distance(coord_w, [alt_axis[b], azi_axis[l]])
                        w_ur = 1. / ia.distance(coord_w, [alt_axis[u], azi_axis[r]])
                        w_br = 1. / ia.distance(coord_w, [alt_axis[b], azi_axis[r]])

                        w_sum = w_ul + w_bl + w_ur + w_br

                        imgs_wrapped[:, ii, jj] = (imgs_raw[:, u, l] * w_ul +
                                                   imgs_raw[:, b, l] * w_bl +
                                                   imgs_raw[:, u, r] * w_ur +
                                                   imgs_raw[:, b, r] * w_br) / w_sum

                        # for cropping
                        if x_min is None:
                            x_min = l;
                            x_max = l + 1;
                            y_min = u;
                            y_max = u + 1
                        else:
                            x_min = min(x_min, l);
                            x_max = max(x_max, l + 1)
                            y_min = min(y_min, u);
                            y_max = max(y_max, u + 1)

        # for testing
        # plt.imshow(img_count, interpolation='bicubic')
        # plt.colorbar()
        # plt.show()

        if is_luminance_correction:
            for frame_ind in range(imgs_wrapped.shape[0]):
                curr_frame = imgs_wrapped[frame_ind]
                curr_mean = np.nanmean(curr_frame.flat)
                curr_frame = curr_frame - curr_mean
                curr_amp = np.max([np.nanmax(curr_frame.flat), abs(np.nanmin(curr_frame.flat))])
                curr_frame = curr_frame / curr_amp
                imgs_wrapped[frame_ind] = curr_frame

        # crop image
        alt_range = np.logical_and(np.arange(imgs_raw.shape[1]) >= y_min,
                                   np.arange(imgs_raw.shape[1]) <= y_max)
        azi_range = np.logical_and(np.arange(imgs_raw.shape[2]) >= x_min,
                                   np.arange(imgs_raw.shape[2]) <= x_max)

        # print imgs_raw.shape
        # print imgs_raw.shape
        # print alt_range.shape
        # print azi_range.shape
        # print np.sum(alt_range)
        # print np.sum(azi_range)

        imgs_dewrapped = imgs_raw[:, alt_range, :]
        imgs_dewrapped = imgs_dewrapped[:, :, azi_range]

        # get degree coordinats of dewrapped images
        deg_coord_alt_ax_dewrapped = alt_axis[alt_range]
        deg_coord_azi_ax_dewrapped = azi_axis[azi_range]
        deg_coord_azi_dewrapped, deg_coord_alt_dewrapped = np.meshgrid(deg_coord_azi_ax_dewrapped,
                                                                       deg_coord_alt_ax_dewrapped)
        deg_coord_alt_dewrapped = deg_coord_alt_dewrapped.astype(np.float32)
        deg_coord_azi_dewrapped = deg_coord_azi_dewrapped.astype(np.float32)

        return imgs_wrapped, self.deg_coord_y, self.deg_coord_x, imgs_dewrapped, deg_coord_alt_dewrapped, \
               deg_coord_azi_dewrapped


class Indicator(object):
    """
    flashing indicator for photodiode

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

    def __init__(self,
                 monitor,
                 width_cm=3.,
                 height_cm=3.,
                 position='northeast',
                 is_sync=True,
                 freq=2.):
        """
        Initialize indicator object
        """

        self.monitor = monitor
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.width_pixel, self.height_pixel = self.get_size_pixel()
        self.position = position
        self.center_width_pixel, self.center_height_pixel = self.get_center()
        self.is_sync = is_sync

        if is_sync == False:
            self.freq = freq  # if not synchronized with stimulation, self update frquency of the indicator
            self.frame_num = self.get_frames()
        else:
            self.freq = None
            self.frame_num = None

    def get_size_pixel(self):

        screen_width = (self.monitor.resolution[1] /
                        self.monitor.downsample_rate)
        screen_height = (self.monitor.resolution[0] /
                         self.monitor.downsample_rate)

        indicator_width = int((self.width_cm / self.monitor.mon_width_cm) *
                              screen_width)
        indicator_height = int((self.height_cm / self.monitor.mon_height_cm) *
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
            raise LookupError('`position` attribute not in '
                              '{"northeast","northwest","southeast","southwest"}.')

        return int(center_width), int(center_height)

    def get_frames(self):
        """
        if not synchronized with stimulation, get frame numbers of each update
        of indicator
        """

        refresh_rate = self.monitor.refresh_rate

        if refresh_rate % self.freq != 0:
            raise ArithmeticError("`freq` not divisble by monitor ref rate.")

        return refresh_rate / self.freq
