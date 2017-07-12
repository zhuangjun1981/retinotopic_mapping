API Documentation
=================

.. toctree::
   :maxdepth: 2

MonitorSetup Module
--------------------------
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

Monitor
+++++++
.. autoclass:: Monitor
   :members: 

Indicator
+++++++++
.. autoclass:: Indicator
   :members:

StimulusRoutines Module
--------------------------

UniformContrast
+++++++++++++++
.. autoclass:: UniformContrast
   :members:

FlashingCircle
++++++++++++++
.. autoclass:: FlashingCircle
   :members:

SparseNoise
+++++++++++
.. autoclass:: SparseNoise
   :members:

DriftingGratingCircle
+++++++++++++++++++++
.. autoclass:: DriftingGratingCircle
   :members:

KSstim
++++++
.. autoclass:: KSstim   
   :members:

KSstimAllDir
++++++++++++
.. autoclass:: KSstimAllDir
   :members:

 

DisplayStimulus Module
--------------------------
Takes care of high level management of your computer 
hardware with respect to its interactions within a given experiment.
Stimulus presentation routines are specified and external connection
to National Instuments hardware devices is provided. Also takes care
of the logging of relevant experimental data collected and where it
will be stored on the computer used for the experiment. 

DisplaySequence 
+++++++++++++++++++++++++

.. autoclass:: DisplaySequence
   :members:







   





