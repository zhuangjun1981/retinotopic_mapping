Installing
==========

Manual install
+++++++++++++++
To manually install **RetinotopicMapping** you can  download the package
`here <https://pypi.python.org/pypi?name=retinotopic-maps&version=2.0.0&:action=display>`_.

Then open the command line, move to the directory that the package was
installed in and run the `setup.py` file like follows:

.. code-block:: python

   cd <full path to the package location>
   python setup.py install


PIP install
++++++++++++
An even simpler way to install the package is with pip. To do so type the following
in the command line:

.. code-block:: python

    pip install retinotopic_maps



Dependencies
+++++++++++++++++++++
        1. numpy, version 1.13.1 or later
        2. scipy, version 0.17.1 or later
        3. matplotlib, version 1.5.1 or later
        4. psychopy, version 1.85.2 or later
        5. pyglet, version 1.2.4 or later
        4. OpenCV-Python, version >= 2.4.8, <= 2.4.10
        5. scikit-image, version 0.12.3 or later
        6. tifffile, version 0.7.0 or later
        7. PyDAQmx, version 1.3.2 or later

        * also requires National Instruments DAQmx driver

Operating Systems Supported
+++++++++++++++++++++++++++

* Windows

* Mac


Making sure PyDAQmx works
+++++++++++++++++++++++++

The PyDAQmx module allows users to integrate National Instruments
data-acquisition hardware into their experimental setup (see their
`documentation <https://pythonhosted.org/PyDAQmx/>`_ for more information).

To get PyDAQmx to function correctly there are a couple of important
points to mention:

* The NIDAQmx driver must first be installed
* Once NIDAQmx driver is installed it is not guaranteed that the
  module will work, so it is important to know how to troubleshoot
  this issue. The main issue is tracking down where two files
  are on your computer, a filepath ending with ``DAQMX.h`` and another
  path ending with ``nicaiu.dll``. The PyDAQmx module tries to find
  these files for you, but if it cannot, the user needs to manually
  find and enter the path within the `DAQmxConfig.py` file.
  See `this <https://pythonhosted.org/PyDAQmx/installation.html>`_
  page for a more thorough explanation).


.. note::
   This is the most likely issue to come up in debugging. Chances
   are if you are having a related issue it either has something
   to do with not supplying a correct path or  making improper
   import statements somewhere in your script.

