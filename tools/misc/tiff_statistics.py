"""
tiff_statistics.py

@author: jayb

Sept 18, 2015
"""
import sys, os
import math
import glob
import cv2
import tifffile
import numpy as np

class TiffStatistics (object):
    ''' Collect statistics about all files in a directory.
        foreach file in directory:
            foreach camera:
                if file matches camera:
                    Extract center region (1Kx1K default) 
                    optionally ignore (but count) the number of files with zero values or full range 14bit (16383) values
                    accumulate statistics Range, Min, Max, Mean, Std, for all center region pixel values

        returns ([stats], [cumulative]) where stats are the per file statistics, and cumulative are the stats for all files in aggregate for each camera
    '''

    def __init__(self, everyNth=1, ignoreSaturated = True, showImages=True, calcHistograms = True, centerSizeX=1024, centerSizeY=1024):
        '''
        everyNth: subsample files in directory
        ignoreSaturated: don't include files with 0 or fully saturated pixels in cumulative results
        showImages: use CV2 to display the center section of each image
        calcHistograms: return the histogram for each image
        centerSizeX, centerSizeY: the center portion of the image to use for analysis in pixels
        '''
        self.everyNth = everyNth
        self.showImages = showImages
        self.calcHistograms = calcHistograms
        self.centerSizeX = centerSizeX
        self.centerSizeY = centerSizeY
        self.ignoreSaturated = ignoreSaturated

    def getStatistics(self, path):
        ''' get statistics for a given directory
        '''
        self.path = path
        if self.showImages:    
            window = cv2.namedWindow("TEST")

        os.chdir(self.path)

        self.images = glob.glob("*.tif")

        # size of region to sample
        x = self.centerSizeX
        y = self.centerSizeY

        hasMin = 0
        hasMax = 0
        maxV = 16383
        stats = []
        count = 0

        for imagePath in self.images:
            count += 1
            if (count % self.everyNth) != 0:
                continue
            with tifffile.TIFFfile(imagePath) as tif:
                frame = tif.asarray()
                w = frame.shape[0]
                h = frame.shape[1]
                sx = (w - x) / 2
                sy = (h - y) / 2
                centerRegion = frame[sy:sy + y, sx: sx + x]
            
                amin = np.amin(centerRegion)
                amax = np.amax(centerRegion)
                mean = np.mean(centerRegion)
                std = np.std(centerRegion)

                hasMin = amin == 0
                hasMax = amax == maxV
                
                s = {'path':imagePath, 'range':amax - amin, 'min':amin, 'max':amax, 'mean':mean, 'std':std, 'hasZero':hasMin, 'hasSaturated':hasMax}
                if self.calcHistograms:
                    s['hist'] = cv2.calcHist(frame, channels=[0], mask=None, histSize=[16384], ranges=[0,16384])

                stats.append(s)

                #out.append( "{:40}: range:{:5d}, min:{:5d}, max:{:5d},
                #mean:{:>6.0f} std:{:>6.0f} {} {}".format(imagePath, amax-amin,
                #amin, amax, mean, std, hasMin, hasMax))

                if self.showImages:
                    cv2.imshow("TEST", centerRegion * 4)
                    k = cv2.waitKey(1)
                    if k == 27:
                        break

        cumulative = []

        class Bunch:
            def __init__(self, **kwds):
                self.__dict__.update(kwds)

        for cam in ['cam0', 'cam1', 'cam2', 'cam3']:
            b = Bunch()
            cumulative.append(b)
            b.cam = cam
            b.hasZero = len([s for s in stats if (cam in s['path'] and s['hasZero'])])
            b.hasSaturated = len([s for s in stats if (cam in s['path'] and s['hasSaturated'])])
            b.count = len([s for s in stats if (cam in s['path'])])
            b.data = [s for s in stats if (cam in s['path'] and not (self.ignoreSaturated and (s['hasZero'] or s['hasSaturated'])))] # skip under and overflows

            if len(b.data):
                t = [row['range'] for row in b.data]
                b.rangeMean = np.mean(t)
                b.rangeMin = np.amin(t)
                b.rangeMax = np.amax(t)

                t = [row['min'] for row in b.data]
                b.minMean = np.mean(t)
                b.minMin = np.amin(t)
                b.minMax = np.amax(t)

                t = [row['max'] for row in b.data]
                b.maxMean = np.mean(t)
                b.maxMin = np.amin(t)
                b.maxMax = np.amax(t)

                t = [row['mean'] for row in b.data]
                b.meanMean = np.mean(t)
                b.meanMin = np.amin(t)
                b.meanMax = np.amax(t)

                t = [row['std'] for row in b.data]
                b.stdMean = np.mean(t)
                b.stdMin = np.amin(t)
                b.stdMax = np.amax(t)

        if self.showImages:
            cv2.destroyAllWindows()

        return (stats, cumulative)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='calculates min, max, mean, std for each file for each camera in directory')

    parser.add_argument("-p", "--path", type=str, help="directory containing tiff files from which to collect statistics",
                        default=os.getcwd())
    parser.add_argument("-e", "--everyNth", type=int, help="only scan every nth file",
                        default=1)
    parser.add_argument("-i", "--ignoreSaturated", type=bool, help="don't include files with 0 or saturated pixels (but include the count of these files)",
                        default=False)
    parser.add_argument("-c", "--calcHistograms", type=bool, help="calc and return a histogram of each file",
                        default=False)
    parser.add_argument("-s", "--showImages", type=bool, help="show images",
                        default=True)
    parser.add_argument("-x", "--centerSizeX", type=bool, help="center X width to analyze",
                        default=1024)    
    parser.add_argument("-y", "--centerSizeY", type=bool, help="center Y height to analyze",
                        default=1024)    
    args = parser.parse_args()

    ts =  TiffStatistics (everyNth=args.everyNth, ignoreSaturated=args.ignoreSaturated, showImages=args.showImages, 
                          calcHistograms = args.calcHistograms, centerSizeX=args.centerSizeX, centerSizeY=args.centerSizeY)

    o = ts.getStatistics(args.path)
    stat = o[0]
    cumulative = o[1]

    print args.path
    print 'Total images: ', len(stat)
    for cam in cumulative:
        if cam.count > 0:
            print "{:4}: zeroCount:{}, saturatedCount:{}, images:{}".format (
                cam.cam, cam.hasZero, cam.hasSaturated, cam.count)
            print "    MIN   range:{:>6.0f},  min:{:>6.0f},  max:{:>6.0f},  mean:{:>6.0f},  std:{:>6.0f}".format (
                cam.rangeMin, cam.minMin, cam.maxMin, cam.meanMin, cam.stdMin)
            print "    MEAN  range:{:>6.0f},  min:{:>6.0f},  max:{:>6.0f},  mean:{:>6.0f},  std:{:>6.0f}".format (
                cam.rangeMean, cam.minMean, cam.maxMean, cam.meanMean, cam.stdMean)
            print "    MAX   range:{:>6.0f},  min:{:>6.0f},  max:{:>6.0f},  mean:{:>6.0f},  std:{:>6.0f}".format (     
                cam.rangeMax, cam.minMax, cam.maxMax, cam.meanMax, cam.stdMax)
            print "    Dynamic range min:  {:.1f} bits".format (math.log(cam.rangeMin, 2))
            print "    Dynamic range mean: {:.1f} bits".format (math.log(cam.rangeMean, 2))
            print "    Dynamic range max:  {:.1f} bits".format (math.log(cam.rangeMax, 2))
            print



