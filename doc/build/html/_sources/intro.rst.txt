Getting Started
===============
The first thing to do after downloading the retinotopic mapping
module is to set up a monitor to display some stimulus routines.
Ideally you should have two monitors, one for running your python 
scripts and another for displaying stimulus. This will allow you 
to get familiar with the code base in a simple and straightforward 
manner, as well as get some practice in debugging your experimental 
setup. 

After working through this tutorial you should be able to display simple 
stimulus routines, and be familiar enough with the basic functionality of the 
experimental part of the code base to debug your own python scripts.

Setting up your monitor
-----------------------
The :mod:`MonitorSetup` module is used to store and keep track of all
your monitor specs as well as the geometry of the experiment and
there are two classes within the  module: :class:`~MonitorSetup.Monitor` 
and :class:`~MonitorSetup.Indicator`. For now we won't pay much attention 
to the :class:`~MonitorSetup.Indicator` class, except to mention that it is 
used for timing purposes. 

Instead we will focus on the :class:`~MonitorSetup.Monitor` class, what it 
does and how it works, but first, here is a visualization of the experimental setup 
to get an idea of what kind of geometerical parameters are stored in the object.

.. image:: images/mouse_geometry.png


The specs you will need to initialize a monitor object are as
follows:

* Monitor resolution 
* Monitor refresh rate
* Monitor width/height in cm.
* Distance from the mouse's eyeball to the monitor in cm
* Distance from the mouse's gaze center to the top of the monitor
* Distance from the mouse's gaze cetner to the anterior edge of the monitor
* Angle between the mouse's body axis and the plane that the monitor lies in.

Since we are just interested in trying to display some stimulus quickly, don't
worry too much about having exact numbers for all of these parameters. It turns
out that ``PsychoPy`` will automatically resize your monitor if you put in the
wrong parameters so just put something in and try to make sure everything
works on your machine.

Let's assume we have a monitor with resolution 1080x1920, refresh rate of 60Hz,
and height and width of 90 and 50 cm respectively, but neglect the geometry
of the experimental setup for now. Then under these assumptions we can initialize
a :class:`~MonitorSetup.Monitor` object as follows

>>> from MonitorSetup import Monitor, Indicator 
>>> mon = Monitor(resolution=(1080,1920),
    	          refresh_rate=60.,
    	          mon_width_cm=90.,
		  mon_height_cm=50.,
		  dis=20,
		  mon_tilt=10.
		  C2T_cm=30.
		  C2A_cm=40.)
>>> ind = Indicator(mon)


Displaying Stimulus
-------------------

Now we are ready to display a stimulus routine on your monitor
from the :mod:`DisplaySequence` class. There are several 
routines to choose from the :mod:`Stimulus` module
(each of which can be previewed :ref:`here <examples>`), but 
for this example we will initialize the :mod:`~StimulusRoutines.FlashingCircle`
routine which takes the ``mon`` and ``ind`` variables 
as parameters.

>>> import StimulusRoutines as stim
>>> from DisplayStimulus import DisplaySequence
>>> flashing_circle = stim.FlashingCircle(mon,ind)
>>> ds = DisplaySequence(log_dir='log_directory')
>>> ds.trigger_display()

which will give an output that should look something like this

.. image:: images/flashing_circle.gif
   :align: center					  
					  

   


