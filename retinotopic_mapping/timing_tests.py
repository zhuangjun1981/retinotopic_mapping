#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:31:10 2017

@author: johnyearsley
"""
import unittest


def time_DriftingGratingCircle_by_index(sf_list=(.08,.16),
                                        tf_list=(4.,8.),
                                        dire_list=(0.,),
                                        con_list=(1.,),
                                        size_list=(10.,),
                                        number=1):
    """Compute amount of time in seconds to generate stimulus movie for 
    DriftingGratingCircle routine when computed by index
    
    Parameters
    ----------
    number : int, optional
        number of times to run the test, can be used to compute average run time.
    
    Returns
    -------
    avg_time : float
        the time, in seconds, it took to generate the stimulus movie.
        
    """
    from timeit import timeit
    
    # write the setup code. you can change the parameters inside of the string
    # for testing purposes
    s = '''
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator

resolution = (1280,1024) 
mon_width_cm = 38 
mon_height_cm = 40
refresh_rate = 60 

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 26.56
dis = 15.

downsample_rate = 4

mon = Monitor(resolution=resolution,
            dis=dis,
            mon_width_cm=mon_width_cm,
            mon_height_cm=mon_height_cm,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
ind = Indicator(mon, position='southwest')

dg = stim.DriftingGratingCircle(mon,
                                ind,
                                sf_list=%r,
                                tf_list=%r,
                                dire_list=%r,
                                con_list=%r,
                                size_list=%r)
'''%(sf_list, tf_list, dire_list, con_list, size_list)

    total_time = timeit('dg.generate_movie_by_index()', setup=s, number=number)
    avg_time = total_time / number
    
    return avg_time

def time_DriftingGratingCircle_non_index(sf_list=(.08,.16),
                                         tf_list=(4.,8.),
                                         dire_list=(0.,),
                                         con_list=(1.,),
                                         size_list=(10.,),
                                         number=1):
    """Compute amount of time in seconds to generate stimulus movie for 
    DriftingGratingCircle routine when computed by index
    
    Parameters
    ----------
    number : int, optional
        number of times to run the test, can be used to compute average run time.
    
    Returns
    -------
    avg_time : float
        the time, in seconds, it took to generate the stimulus movie.
        
    """
    from timeit import timeit
    
    # write the setup code. you can change the parameters inside of the string
    # for testing purposes
    s = '''
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator

resolution = (1280,1024) 
mon_width_cm = 38 
mon_height_cm = 40
refresh_rate = 60 

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 26.56
dis = 15.

downsample_rate = 4

mon = Monitor(resolution=resolution,
            dis=dis,
            mon_width_cm=mon_width_cm,
            mon_height_cm=mon_height_cm,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
ind = Indicator(mon, position='southwest')

dg = stim.DriftingGratingCircle(mon,
                                ind,
                                sf_list=%r,
                                tf_list=%r,
                                dire_list=%r,
                                con_list=%r,
                                size_list=%r)
'''%(sf_list, tf_list, dire_list, con_list, size_list)

    total_time = timeit('dg.generate_movie()', setup=s, number=number)
    avg_time = total_time / number
    
    return avg_time

def time_SparseNoise_by_index(number=1):
    """ Compute amount of time, in seconds, to generate frames by index for 
    SparseNoise stimulus routine.
    
    Parameters
    ----------
    number : int, optional
        number of times to run test, can be used for finding avergae run time.
        
    Returns
    -------
    avg_time : float
        average amount of time to generate the stimulus movie
        
    """
    
    from timeit import timeit
    
    s = '''
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator

resolution = (1280,1024) 
mon_width_cm = 38 
mon_height_cm = 40
refresh_rate = 60 

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 26.56
dis = 15.

downsample_rate = 4

mon = Monitor(resolution=resolution,
            dis=dis,
            mon_width_cm=mon_width_cm,
            mon_height_cm=mon_height_cm,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
ind = Indicator(mon, position='southwest')

sn = stim.SparseNoise(mon,ind,iteration=2,probe_frame_num=3)
'''

    total_time = timeit('sn.generate_movie_by_index()', setup=s, number=number)
    avg_time = total_time / number
    
    return avg_time

def time_SparseNoise_non_index(number=1):
    """ Compute amount of time, in seconds, to generate frames by index for 
    SparseNoise stimulus routine.
    
    Parameters
    ----------
    number : int, optional
        number of times to run test, can be used for finding avergae run time.
        
    Returns
    -------
    avg_time : float
        average amount of time to generate the stimulus movie
        
    """
    
    from timeit import timeit
    
    s = '''
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator

resolution = (1280,1024) 
mon_width_cm = 38 
mon_height_cm = 40
refresh_rate = 60 

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 26.56
dis = 15.

downsample_rate = 4

mon = Monitor(resolution=resolution,
            dis=dis,
            mon_width_cm=mon_width_cm,
            mon_height_cm=mon_height_cm,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
ind = Indicator(mon, position='southwest')

sn = stim.SparseNoise(mon,ind,iteration=2,probe_frame_num=3)
'''

    total_time = timeit('sn.generate_movie()', setup=s, number=number)
    avg_time = total_time / number
    
    return avg_time

if __name__ == '__main__':
    print '\n===== Running Tests ====='
    unittest.main(verbosity=2)
    
    # To test display by index routines for DriftingGratingCircle
    # change functions parameters inside the 'time_' functions
    time_by_index_DG = time_DriftingGratingCircle_by_index()
    time_non_index_DG = time_DriftingGratingCircle_non_index()
    speedup_DG = time_non_index_DG / time_by_index_DG
    
    time_by_index_SN = time_SparseNoise_by_index()
    time_non_index_SN = time_SparseNoise_non_index()
    speedup_SN = time_by_index_SN / time_non_index_SN
    
    print 'DriftingGrating index routine  : %r seconds' % round(time_by_index_DG,2)
    print 'DriftingGrating non index routine : %r seconds' % round(time_non_index_DG,2)
    print 'Amount of speedup : %r x' % round(speedup_DG,2)
    
    print 'SparseNoise index routine : %r seconds' % round(time_by_index_SN,2)
    print 'SparseNoise non index routine : %r seconds' %round(time_non_index_SN,2)
    print 'Amount of speedup : %r x' % round(speedup_SN,2)
    
    

