# -*- coding: utf-8 -*-
"""
Example script to test that everything is working. Running this script is a 
good first step for trying to debug your experimental setup.

!!!IMPORTANT!!!
Note that once you are displaying stimulus, if you wan't to stop the code from
running all you need to do is press either one of the 'Esc' or 'q' buttons.

"""
import Stimulus as stim
from MonitorSetup import Monitor, Indicator
from DisplayStimulus import DisplaySequence 

"""
To get up and running quickly before performing any experiments it is 
sufficient to setup two monitors -- one for display and one for your python 
environment. If you don't have two monitors at the moment it is doable with
only one. 

Uncomment the following block after you have entered the respective parameters.
Since this script is for general debugging and playing around with the code, 
we will arbitrarily populate variables that describe the geometry of where 
the mouse will be located during an experiment. All we are interested in 
here is just making sure that we can display stimulus on a monitor.
"""
#resolution = (  ) #enter your monitors resolution
#mon_width_cm = _ #enter your monitors width in cm
#mon_height_cm = _ #enter your monitors height in cm
#refresh_rate = _  #enter your monitors height in Hz

"""The following variables correspond to the geometry of the mouse with 
respect to the monitor, don't worry about them for now"""
C2T_cm = 31.1
C2A_cm = 41.91
mon_tilt = 26.56
dis = 13.5

downsample_rate = _

mon=Monitor(resolution=(1080, 1920),
            dis=dis,
            mon_width_cm=mon_width,
            mon_height_cm=mon_tilt,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
indicator=Indicator(mon)

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=5)
#indicator=Indicator(mon)
#KS_stim=stim.KSstim(mon,indicator)
#display_iter = 2
## print (len(KSstim.generate_frames())*display_iter)/float(mon.refresh_rate)
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=r'C:\data',is_triggered=True,display_iter=2,display_screen=1)
#ds.set_stim(KS_stim)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

   
#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=10)
#indicator=Indicator(mon)
#flashing_circle=FlashingCircle(mon,indicator)
#display_iter = 2
#print (len(flashing_circle.generate_frames())*display_iter)/float(mon.refresh_rate)
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=r'C:\data',is_triggered=True,display_iter=2,display_screen=1)
#ds.set_stim(flashing_circle)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#indicator=Indicator(mon)
#sparse_noise=SparseNoise(mon,indicator, subregion=(-20.,20.,40.,60.), grid_space=(10, 10))
#grid_points = sparse_noise._generate_grid_points_sequence()
#gridLocations = np.array([l[0] for l in grid_points])
#plt.plot(mon_points[:,0],mon_points[:,1],'or',mec='#ff0000',mfc='none')
#plt.plot(gridLocations[:,0], gridLocations[:,1],'.k')
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#indicator=Indicator(mon)
#sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
#grid_points = sparse_noise._generate_grid_points_sequence()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=20)
#mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#indicator=Indicator(mon)
#sparse_noise=SparseNoise(mon,indicator,subregion=(-20.,20.,40.,60.))
#sparse_noise.generate_frames()
#==============================================================================================================================

#==============================================================================================================================
#mon = Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#frame = get_warped_square(mon.deg_coord_x,mon.deg_coord_y,(20.,25.),4.,4.,0.,foreground_color=1,background_color=0)
#plt.imshow(frame,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#mon_points = np.transpose(np.array([mon.deg_coord_x.flatten(),mon.deg_coord_y.flatten()]))
#indicator=Indicator(mon)
#sparse_noise=SparseNoise(mon,indicator)
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
#ds.set_stim(sparse_noise)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=20)
#indicator=Indicator(mon)
#KS_stim_all_dir=KSstimAllDir(mon,indicator,step_width=0.3)
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter = 2,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
#ds.set_stim(KS_stim_all_dir)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1080, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=16.22,downsample_rate=5)
#indicator=Indicator(mon)
#    
#grating = get_grating(mon.deg_coord_x, mon.deg_coord_y, ori=0., spatial_freq=0.1, center=(60.,0.), contrast=1)
#print grating.max()
#print grating.min()
#plt.imshow(grating,cmap='gray',interpolation='nearest',vmin=0., vmax=1.)
#plt.show()
#    
#drifting_grating = DriftingGratingCircle(mon,indicator, sf_list=(0.08,0.16),
#                                         tf_list=(4.,8.), dire_list=(0.,0.1),
#                                         con_list=(0.5,1.), size_list=(5.,10.),)
#print '\n'.join([str(cond) for cond in drifting_grating._generate_all_conditions()])
#    
#drifting_grating2 = DriftingGratingCircle(mon,indicator,
#                                          center=(60.,0.),
#                                          sf_list=[0.08, 0.16],
#                                          tf_list=[4.,2.],
#                                          dire_list=[np.pi/6],
#                                          con_list=[1.,0.5],
#                                          size_list=[40.],
#                                          block_dur=2.,
#                                          pregap_dur=2.,
#                                          postgap_dur=3.,
#                                          midgap_dur=1.)
#frames =  drifting_grating2.generate_frames()
#print '\n'.join([str(frame) for frame in frames])
#    
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter = 2,is_triggered=False,is_sync_pulse_pulse=False,is_interpolate=False,display_screen=1)
#ds.set_stim(drifting_grating2)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon=Monitor(resolution=(1200, 1920),dis=13.5,mon_width_cm=88.8,mon_height_cm=50.1,C2T_cm=33.1,C2A_cm=46.4,mon_tilt=30,downsample_rate=5)
#indicator=Indicator(mon)
#uniform_contrast = UniformContrast(mon,indicator, duration=10., color=0.)
#ds=DisplaySequence(log_dir=r'C:\data',backupdir=None,display_iter=2,is_triggered=False,is_sync_pulse_pulse=False,display_screen=1)
#ds.set_stim(uniform_contrast)
#ds.trigger_display()
#plt.show()
#==============================================================================================================================

#==============================================================================================================================
#mon = Monitor(resolution=(1080, 1920), dis=13.5, mon_width_cm=88.8, mon_height_cm=50.1, C2T_cm=33.1, C2A_cm=46.4, mon_tilt=16.22,
#   downsample_rate=5)
#indicator = Indicator(mon)
#drifting_grating2 = DriftingGratingCircle(mon, indicator,
#                          center=(60., 0.),
#                          sf_list=[0.08],
#                          tf_list=[4.],
#                          dire_list=np.arange(0, 2 * np.pi, np.pi / 4),
#                          con_list=[1.],
#                          size_list=[20.],
#                          block_dur=2.,
#                          pregap_dur=2.,
#                          postgap_dur=3.,
#                          midgap_dur=1.)
#ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, display_iter=1, is_triggered=True, is_sync_pulse_pulse=True,
#          is_interpolate=False,display_screen=1)
#ds.set_stim(drifting_grating2)
#ds.trigger_display()
#=============================================================================#