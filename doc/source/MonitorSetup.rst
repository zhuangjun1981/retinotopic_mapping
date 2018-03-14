retinotopic_mapping.MonitorSetup
================================
Used to store the display monitor and particular geometry used within a given
experimental setup. The `Monitor` class holds references to the sizing of the
monitor that is used to display stimulus routines and contains the necessary
geometrical description of where the subject's eye is placed with respect to the
display monitor. The `Indicator` class, on the other hand, is generally used in
order to gain precise temporal information of the stimuli display. Basically a 
indicator is just a small square shows up at one corner of the display screen. 
It changes color during visual stimuli display. By covering it with a photodiode 
with high temporal precision and syncing the photodiode signal into data 
acquisition system, the experimenter will get the ground truth of display 
timestamps. Ideally the flashes of an indicator will be synchronized with the 
onsets of stimuli.

The module will most definitely be used in conjunction with the :mod:`DisplayStimulus`
and :mod:`StimulusRoutines` modules.

Monitor
-------
.. autoclass:: retinotopic_mapping.MonitorSetup.Monitor
   :members:


Indicator
---------
.. autoclass:: retinotopic_mapping.MonitorSetup.Indicator
   :members:
