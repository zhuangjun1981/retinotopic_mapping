About
=====
The retinotopic mapping package is a self-contained module
for performing automated segmentation of the mouse
visual cortex. The experimental setup and analysis routine was
modified from Garrett et al. 2014 (1), and closely follows
the protocols and procedures documented in Juavinett et al. 2016
(2). 

The code base contains several stimulus routines which are 
highly customizable and designed to give the user significant
flexibility and control in creative experimental design. There
are two distinct but connected aspects to the package:

1. an online experimental component comprised of the 
`MonitorSetup`, `StimulusRoutines`, and `DisplayStimulus` modules

2. an offline automated analysis component provided
by the `RetinotopicMapping` module



.. image:: images/vasculature.png

What is a retinotopic map?
++++++++++++++++++++++++++
Retinotopic maps are a common tools used in systems 
neuroscience to understand how receptive fields are
mapped onto particular regions of the brain. In the lower visual
areas of certain species neurons are organized as a 2D representation 
of the visual image formed on the retina wherein neighboring regions 
of a particular image are represented by neighboring regions of 
the visual area.

.. image:: images/retinotopic_maps.png
   :scale: 60%


Contributors
~~~~~~~~~~~~
* Jun Zhuang @zhuang1981
* John Yearsley @yearsj
* Derric Williams @derricw
