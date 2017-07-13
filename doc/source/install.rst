Installing
==========
To install **RetinotopicMapping** download the package (insert link here)
manually and from the command line type in the following:

.. code-block:: python

   cd <path to package>
   python setup.py install

or with pip 

.. code-block:: python
    
    pip install retinotopic_maps



Dependencies
+++++++++++++++++++++
        1. numpy, version 1.10.4 or later
        2. scipy, version 0.17.0 or later
        3. matplotlib, version 1.5.1 or later
        4. OpenCV-Python, version 2.4.8 or later
        5. scikit-image, version 0.12.3 or later
        6. tifffile, version 0.7.0 or later
	7. PyDAQmx, version ...

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



