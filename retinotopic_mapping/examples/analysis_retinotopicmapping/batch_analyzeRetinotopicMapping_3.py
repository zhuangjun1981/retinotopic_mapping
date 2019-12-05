import os
import scipy.ndimage as ni
import matplotlib.pyplot as plt
import corticalmapping.RetinotopicMapping as rm
import corticalmapping.core.PlottingTools as pt

trialName = "180308_M360495_Trial1.pkl"
isPlotName = False
isSave = True

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trial, _ = rm.loadTrial(trialName)

if hasattr(trial, 'finalPatchesMarked'):
    patches_to_show = trial.finalPatchesMarked
elif hasattr(trial, 'finalPatches'):
    patches_to_show = trial.finalPatches
else:
    raise LookupError('cannot find finalPatches in the trial.')

vasmap = trial.vasculatureMap

f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.imshow(vasmap, vmin=0., vmax=1., cmap='gray', interpolation='nearest')


for patch_n, patch in patches_to_show.items():

    if patch.sign == 1:
        color = '#ff0000'
    elif patch.sign == -1:
        color = '#0000ff'
    else:
        color = '#000000'

    zoom = float(vasmap.shape[0]) / patch.array.shape[0]

    mask = ni.binary_erosion(patch.array, iterations=1)
    pt.plot_mask_borders(mask, plotAxis=ax, color=color, zoom=zoom, borderWidth=2)

    if isPlotName:
        cen = patch.getCenter()
        ax.text(cen[1] * zoom, cen[0] * zoom, patch_n, va='center', ha='center', color=color)

ax.set_axis_off()
plt.show()

if isSave:
    pt.save_figure_without_borders(f, '{}_AreaBorders.pdf'.format(trial.getName()), dpi=300)
    pt.save_figure_without_borders(f, '{}_AreaBorders.png'.format(trial.getName()), dpi=300)