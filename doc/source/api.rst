API Documentation
=================

.. toctree::
   :maxdepth: 2

MonitorSetup
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
.. autoclass:: retinotopic_mapping.MonitorSetup.Monitor
   :members: 

Indicator
+++++++++
.. autoclass:: retinotopic_mapping.MonitorSetup.Indicator
   :members:


StimulusRoutines
------------------

UniformContrast
+++++++++++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.UniformContrast
   :members:

FlashingCircle
++++++++++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.FlashingCircle
   :members:

SparseNoise
+++++++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.SparseNoise
   :members:

DriftingGratingCircle
+++++++++++++++++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.DriftingGratingCircle   
   :members:

KSstim
+++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.KSstim
   :members:

KSstimAllDir
+++++++++++++
.. autoclass:: retinotopic_mapping.StimulusRoutines.KSstimAllDir
   :members:



DisplayStimulus
----------------

DisplaySequence
++++++++++++++++
.. autoclass:: retinotopic_mapping.DisplayStimulus.DisplaySequence
   :members:

