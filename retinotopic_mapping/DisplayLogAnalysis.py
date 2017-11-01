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

