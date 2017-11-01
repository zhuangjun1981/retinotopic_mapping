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

    def get_stimuli_type_sequence(self):
        pass
