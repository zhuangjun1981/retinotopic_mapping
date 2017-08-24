__author__ = 'junz'

import os
import matplotlib.pyplot as plt
import retinotopic_mapping.RetinotopicMapping as rm
from tools import FileTools as ft


trialName = "160211_M214522_Trial1.pkl"
isSave = True

params = {'phaseMapFilterSigma': 1.,
          'signMapFilterSigma': 9.,
          'signMapThr': 0.3,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 10.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trial = rm.loadTrial(trialName)

trial.params=params

trial.processTrial(isPlot=True)

trialDict = trial.generateTrialDict()
trial.plotTrial(isSave=isSave,saveFolder=currFolder)
plt.show()

if isSave:
    ft.saveFile(trial.getName()+'.pkl',trialDict)



