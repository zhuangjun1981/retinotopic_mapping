# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:46:38 2014

@author: junz
"""
import os
import matplotlib.pyplot as plt
import corticalmapping.core.FileTools as ft
import corticalmapping.RetinotopicMapping as rm


trialName = '160208_M193206_Trial1.pkl'

names = [
         ['patch01', 'V1'],
         ['patch02', 'RL'],
         ['patch03', 'LM'],
         ['patch04', 'AL'],
         ['patch05', 'AM'],
         ['patch06', 'PM'],
         ['patch07', 'MMA'],
         ['patch08', 'MMP'],
         ['patch09', 'LLA'],
         # ['patch10', 'AM'],
         # ['patch11', 'LLA'],
         # ['patch12', 'MMP'],
         # ['patch13', 'MMP']
         # ['patch14', 'MMP']
         ]

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trialPath = os.path.join(currFolder,trialName)

trialDict = ft.loadFile(trialPath)

finalPatches = dict(trialDict['finalPatches'])

for i, namePair in enumerate(names):
    currPatch = finalPatches.pop(namePair[0])
    newPatchDict = {namePair[1]:currPatch}
    finalPatches.update(newPatchDict)
    
trialDict.update({'finalPatchesMarked':finalPatches})

ft.saveFile(trialPath,trialDict)

trial, _ = rm.loadTrial(trialPath)
f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)
trial.plotFinalPatchBorders(plotAxis = ax,borderWidth=4)
plt.show()
f.savefig(trialName[0:-4]+'_borders.pdf',dpi=600)
f.savefig(trialName[0:-4]+'_borders.png',dpi=300)