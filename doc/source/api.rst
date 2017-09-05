API Documentation
=================

.. toctree::
   :maxdepth: 4

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
.. automodule:: retinotopic_mapping.MonitorSetup
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
Visual Stimulus codebase implements several classes to display stimulus routines.
Can display frame by frame or compress data for certain stimulus routines and
display by index. Used to manage information between experimental devices and
interact with :mod:`StimulusRoutines` module to produce visual display and log data.
May also be used to save and export movies of experimental stimulus routines for
presentation.

DisplaySequence
++++++++++++++++
.. autoclass:: retinotopic_mapping.DisplayStimulus.DisplaySequence
   :members:

