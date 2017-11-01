'''
2017-10-31 by Jun Zhuang
this module provides analysis tools to extract information about visual stimuli
saved in the log pkl files.
'''

import numpy as np
import tools.FileTools as ft

class DisplayLogAnalyzer(object):

    def __init__(self, log_path):

        self.log_dict = ft.loadFile(log_path)

        if not self.log_dict['presentation']['is_by_index']:
            raise NotImplementedError('The visual stimuli display should be indexed.')

        self.check_integrity()

    def check_integrity(self):

        print(self.log_dict['presentation']['frame_stats'])

        if not self.log_dict['presentation']['keep_display']:
            raise ValueError('Stimulus presentation did not end normally.')

        total_frame1 = len(self.log_dict['presentation']['displayed_frames'])
        total_frame2 = len(self.log_dict['presentation']['frame_ts_start'])
        total_frame3 = len(self.log_dict['presentation']['frame_ts_end'])
        total_frame4 = len(self.log_dict['stimulation']['index_to_display'])
        if not total_frame1 == total_frame2 == total_frame3 == total_frame4:
            print('\nNumber of displayed frames: {}.'.format(total_frame1))
            print('\nNumber of frame start timestamps: {}.'.format(total_frame2))
            print('\nNumber of frame end timestamps: {}.'.format(total_frame3))
            print('\nNumber of frames to be displayed: {}.'.format(total_frame4))
            raise ValueError('Numbers of total frames do not agree with each other from various places.')

        if max(self.log_dict['stimulation']['index_to_display']) >= \
                len(self.log_dict['stimulation']['frames_unique']):
            raise ValueError('Display index beyond number of unique frames.')

    @property
    def num_frame_tot(self):
        return len(self.log_dict['presentation']['displayed_frames'])


