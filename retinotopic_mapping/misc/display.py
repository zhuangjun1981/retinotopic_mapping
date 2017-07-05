"""
A few utilities for displaying images.

DisplayWindow is simply an OpenCV window with some added features like:
    - multiple mouse event callbacks (normally the window only supports one)
    - keyboard callbacks
    - image scaling

MiniMap is a DisplayWindow designed for building a tiled map out of several
    images.

"""

import numpy as np
import cv2
# cv3 has mousewheel support
# cv2 does not
if cv2.__version__ >= "3":
    MOUSEWHEEL = cv2.EVENT_MOUSEWHEEL
elif cv2.__version__ < "3":
    MOUSEWHEEL = False


class DisplayWindow(object):
    """
    OpenCV display window with some extra features.

        - Scales its output before displaying
        - Can add/remove crosshairs
        - Keybinds
        - Mouse event callbacks

    Args:
        name (str): name of the window.  cannot have two windows with same
            name at the same time.

    Example:

        >>> win = DisplayWindow("my_window")
        >>> win.scale = 2.0
        >>> win.crosshairs = True
        >>> win.show(img)
        >>> win.close()

    """
    def __init__(self, name="display"):
        super(DisplayWindow, self).__init__()
        self.crosshairs = False
        self.crosshair_color = 0.0
        self._name = name
        
        self._scale = 1.0
        self._current_image = None
        self._scaled_image = None
        self._key_callbacks = []
        self._mouse_callbacks = []
        self._keybinds = {}
        cv2.namedWindow(self._name)

        # set some default keybinds and mouse events
        self.set_key_callback(self._keybind)
        cv2.setMouseCallback(self.name, self._mouse_event)
        self.set_mouse_callback(self._mouse_wheel)

        self._set_default_keybinds()

    @property
    def name(self):
        return self._name
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        if value > 0:
            self._scale = value

    def scale_up(self, increment=0.01):
        self.scale += increment

    def scale_down(self, increment=0.01):
        self.scale -= increment

    def _rescale(self, img):
        if self._scaled_image is not None:
            scaled_size = self._scaled_image.shape
            origin_size = self._current_image.shape

            if self._scaled_image.ndim == self._current_image.ndim:

                if (scaled_size[0] == int(origin_size[0]*self.scale)) and \
                   (scaled_size[1] == int(origin_size[1]*self.scale)):
                    # don't re-allocate if size is identical to last time
                    self._scaled_image[:] = cv2.resize(img,
                                                       dsize=(0,0),
                                                       fx=self.scale,
                                                       fy=self.scale)
                    return
        self._scaled_image = cv2.resize(img,
                                        dsize=(0,0),
                                        fx=self.scale,
                                        fy=self.scale)

    def show(self, img=None):
        """
        Shows an image.

        Args:
            img (numpy.ndarray): any image that you could show in an opencv
                window
        """
        if img is not None:
            self._current_image = img
        if self._current_image is not None:
            if self.scale == 1.0:
                to_show = self._current_image
            else:
                self._rescale(self._current_image)
                to_show = self._scaled_image
            if self.crosshairs:
                to_show[:] = self._draw_crosshairs(to_show)
            cv2.imshow(self.name, to_show)
            self.update()
        else:
            raise ValueError("No image to show.")

    def save_image(self, path):
        """ Save current image to path using opencv.
        """
        cv2.imsave(path, self._current_image)

    def _draw_crosshairs(self, img):
        img_shape = img.shape
        y, x = int(img_shape[0]/2), int(img_shape[1]/2)
        img[y-1:y+1] = self.crosshair_color
        img[:,x-1:x+1] = self.crosshair_color
        return img

    def update(self):
        """
        Handles events like keystrokes, etc.  Should be run periodically.

        *Is run automatically by `show()`*
        """
        k = cv2.waitKey(1)
        if k is not -1:
            self._key_down(k)

    def set_key_callback(self, callback):
        """
        Adds a keystroke callback. After an `update` if a key was pressed, all
            callbacks will be called with they keycode as the argument.

        Args:
            callback (callable): function to call after a keystroke

        """
        self._key_callbacks.append(callback)

    def set_keybind(self, key, callback):
        """
        Sets a keybind to a specific key.

        Args:
            key (str): single character to bind
            callback (callable): function called when key is pressed

        """
        if isinstance(key, int):
            self._keybinds[key] = callback
        elif isinstance(key, str):
            self._keybinds[ord(key)] = callback
        else:
            raise TypeError("Key type should be str or int.")

    def _set_default_keybinds(self):
        self.set_keybind("=", self.scale_up)
        self.set_keybind("-", self.scale_down)

    def set_mouse_callback(self, callback):
        """
        Registers a mouse event callback.

        Args:
            callback (callable): function for mouse event. args are
                (event, x, y, flags, param)
        """
        self._mouse_callbacks.append(callback)

    def _key_down(self, k):
        for callback in self._key_callbacks:
            callback(k)

    def _keybind(self, k):
        if k in self._keybinds:
            self._keybinds[k]()

    def _mouse_event(self, event, x, y, flags, param):
        for callback in self._mouse_callbacks:
            callback(event, x, y, flags, param)

    def _mouse_wheel(self, event, x, y, flags, param):
        if not MOUSEWHEEL:
            pass
        else:
            if event == MOUSEWHEEL:
                if flags > 0:
                    self.scale_up()
                else:
                    self.scale_down()

    def set_pos(self, x, y):
        """
        Sets the window position.

        Args:
            x (int): x position in monitor coords
            y (int): y position in monitor coords
        """
        self.update()
        cv2.moveWindow(self.name, x, y)
        self.update()

    def close(self):
        cv2.destroyWindow(self.name)


class MiniMap(object):
    """
    An image built out of downsampled tiled images.  Optionally displays the
        map in an opencv window.

    Args:
        overlap (float): how much to overlap the images. (0.0->1.0)
        downsample_factor (int): how much to downsample the image tiles default
            is 10X.
        display (bool): whether to display the windows

    Example:
        >>> mm = MiniMap(overlap=0.10)
        >>> mm.add_image(img, 0, 0)  # adds an image at position (0, 0)
        >>> mm.add_image(img, 0, 1)  # adds an image at position (0, 1)
        >>> mm.clear()  # clears the minimap
        
    """
    def __init__(self,
                 overlap=0.10,
                 downsample_factor=10,
                 display=True):
        super(MiniMap, self).__init__()
        self.grid_shape = (1,1)
        self.scale_factor = (downsample_factor,downsample_factor)
        self.image_shape = None
        self.map_shape = None
        
        if 0 <= overlap < 1.0:
            self._overlap = overlap
        else:
            raise ValueError("Overlap out of bounds.")

        self.map = None
        self.map_shape = None
        self._xmin = 0
        self._ymin = 0

        self._images = []

        if display:
            self.display = DisplayWindow(name="map")
        else:
            self.display = None

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        # todo: should we allow changes to overlap?
        self.clear()
        self._overlap = value

    def _resize_map(self, dy=0, dx=0):
        """
        Resizes the map in x or y.
        """
        if self.map is not None:
            old_map = self.map[:]
            old_map_shape = self.map_shape
        y, x = self.grid_shape
        self.grid_shape = (y + abs(dy), x + abs(dx))
        ypix, xpix = self._get_map_size()
        self.map = np.zeros((ypix, xpix), dtype=np.uint8)
        self.map_shape = ypix, xpix
        if dy < 0:
            #move map down
            self._ymin += dy
            self.map[ypix-old_map_shape[0]:,:] = old_map
        elif dy > 0:
            #resize map
            self.map[0:old_map_shape[0],:] = old_map
        if dx < 0:
            #move map right
            self._xmin += dx
            #resize map
            self.map[:, xpix-old_map_shape[1]:] = old_map
        elif dx > 0:
            #resize map
            self.map[:, 0:old_map_shape[1]] = old_map

    def _get_map_size(self):
        """
        Returns the size of the map in pixels.
        """
        dsy, dsx = self.ds_shape
        dsyo, dsxo = int(dsy * (1 - self.overlap)), int(dsx * (1 - self.overlap))

        ypix, xpix = (dsy + (self.grid_shape[0] - 1)*dsyo,
                      dsx + (self.grid_shape[1] - 1)*dsxo)
        return ypix, xpix

    def _get_pixel_location(self, xpos, ypos):
        """
        Converts tile position to map pixel location, accounting for overlap.
        """
        dsy, dsx = self.ds_shape
        dsyo, dsxo = int(dsy * (1 - self.overlap)), int(dsx * (1 - self.overlap))
        yoffset = abs(self._ymin)*dsyo + ypos*dsyo
        xoffset = abs(self._xmin)*dsxo + xpos*dsxo
        return yoffset, xoffset


    def clear(self):
        """
        Clears the map.
        """
        self.map = None
        self._images = []
        self.image_shape = None
        self.grid_shape = (1, 1)

    def _downsample_image(self, img):
        """
        Downsamples the image.
        """
        yscale, xscale = self.scale_factor
        downsampled = img[::yscale, ::xscale]
        self.ds_shape = downsampled.shape
        return downsampled

    def add_image(self, img, xpos, ypos):
        """
        Adds an image to the map at the specified grid location.
        """
        #print "Adding image at {} {}".format(ypos, xpos)
        
        downsampled = self._downsample_image(img)
        ygrid, xgrid = self.grid_shape

        if self.image_shape is None:
            self.image_shape = img.shape
            self._resize_map()

        dsy, dsx = self.ds_shape
        
        yoffset, xoffset = self._get_pixel_location(xpos, ypos)

        yend, xend = yoffset + dsy, xoffset + dsx

        # print yoffset, xoffset, yend, xend
        # print self.map.shape
        # print self._ymin, self._xmin

        if yoffset < 0:
            self._resize_map(dy=-1)
            self.add_image(img, xpos, ypos)
        elif xoffset < 0:
            self._resize_map(dx=-1)
            self.add_image(img, xpos, ypos)
        elif yend > self.map.shape[0]:
            self._resize_map(dy=1)
            self.add_image(img, xpos, ypos)
        elif xend > self.map.shape[1]:
            self._resize_map(dx=1)
            self.add_image(img, xpos, ypos)
        else:
            self._paste_img(downsampled, yoffset, xoffset)

            self._images.append((downsampled, xpos, ypos))

        if self.display:
            self.display.show(self.map)

    def _paste_img(self, downsampled, yoffset, xoffset, blend=False):
        """
        Adds a downsampled image to the map.
        """
        dsy, dsx = downsampled.shape
        if not blend:
            self.map[yoffset:yoffset + dsy,
                     xoffset:xoffset + dsx] = downsampled
        else:
            raise NotImplementedError("Tile blending is not implemented yet.")

    def update(self):
        """
        Updates the display window, handles keypresses.
        """
        self.display.update()
        try:
            self.display.show()
        except ValueError:
            pass


if __name__ == '__main__':
    
    import sys

    img_path = r"C:\Users\derricw\Pictures\steamcat.jpg"
    img = cv2.imread(img_path)

    d = DisplayWindow(name="test")
    d.scale = 1.0
    d.crosshairs = True
    d.show(img)

    d.set_keybind("q", exit)

    while True:
        d.show()
        d.update()

    d.close()