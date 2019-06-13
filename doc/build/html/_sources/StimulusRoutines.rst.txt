retinotopic_mapping.StimulusRoutines
====================================
This module define a base class :class:`Stim` for visual stimulation. 
Each subclass of :class:`Stim` defines a particular type of visual 
stimulus, i.e. :class:`UniformContrast` or :class:`SparseNoise`. When 
initiated, these subclasses take various parameter inputs to generate  
stimulus arrays and metadata dictionary which can be passed to the 
:class:`DisplayStimulus.DisplaySequence` for displaying. Each subclass 
will have a method called `generate_movie()` or 
`generate_movie_by_index()` or both. Only when these methods are 
called, will heavy lifting calculation take place. 

Stim
----
.. autoclass:: retinotopic_mapping.StimulusRoutines.Stim


UniformContrast
---------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.UniformContrast
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


FlashingCircle
--------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.FlashingCircle
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


SparseNoise
-----------
.. autoclass:: retinotopic_mapping.StimulusRoutines.SparseNoise
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


LocallySparseNoise
------------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.LocallySparseNoise
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


DriftingGratingCircle
---------------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.DriftingGratingCircle
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


StaticGratingCircle
-------------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.StaticGratingCircle
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


StaticImages
------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.StaticImages
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


StimulusSeparator
-----------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.StimulusSeparator
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


CombinedStimuli
---------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.CombinedStimuli
   :members:


KSstim
------
.. autoclass:: retinotopic_mapping.StimulusRoutines.KSstim
   :inherited-members: retinotopic_mapping.StimulusRoutines.Stim
   :members:


KSstimAllDir
------------
.. autoclass:: retinotopic_mapping.StimulusRoutines.KSstimAllDir
   :members:


