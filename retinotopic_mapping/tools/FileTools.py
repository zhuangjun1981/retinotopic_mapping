__author__ = 'junz'

import numpy as np
import pickle
import os
import shutil
import ImageAnalysis as ia
import tifffile as tf


try:
     import cv2
except ImportError as e: print 'can not import OpenCV. ' + e


def saveFile(path,data):
    f = open(path,'wb')
    pickle.dump(data, f)
    f.close()


def loadFile(path):
    f = open(path,'rb')
    data = pickle.load(f)
    f.close()
    return data


def copy(src, dest):
    '''
    copy everything from one path to another path. Work for both direcory and file.
    if src is a file, it will be copied into dest
    if src is a directory, the dest will have the same content as src
    '''

    if os.path.isfile(src):
        print 'Source is a file. Starting copy...'
        try: shutil.copy(src,dest); print 'End of copy.'
        except Exception as error: print error

    elif os.path.isdir(src):
        print 'Source is a directory. Starting copy...'
        try: shutil.copytree(src, dest); print 'End of copy.'
        except Exception as error: print error

    else: raise IOError, 'Source is neither a file or a directory. Can not be copied!'


def list_all_files(folder):
    '''
    get a list of full path of all files in a folder (including subfolder)
    '''
    files = []
    for folder_path, subfolder_paths, file_names in os.walk(folder):
        for file_name in file_names:
            files.append(os.path.join(folder_path,file_name))
    return files


def batchCopy(pathList, destinationFolder, isDelete=False):
    '''
    copy everything in the pathList into destinationFolder
    return a list of paths which can not be copied.
    '''

    if not os.path.isdir(destinationFolder): os.mkdir(destinationFolder)

    unCopied=[]

    for path in pathList:
        print '\nStart copying '+path+' ...'
        if os.path.isfile(path):
            print 'This path is a file. Keep copying ...'
            try:
                shutil.copy(path,destinationFolder)
                print 'End of copying.'
                if isDelete:
                    print 'Deleting this file ...'
                    try: os.remove(path); print 'End of deleting.\n'
                    except Exception as error: print 'Can not delete this file.\nError message:\n'+str(error)+'\n'
                else: print ''
            except Exception as error: unCopied.append(path);print 'Can not copy this file.\nError message:\n'+str(error)+'\n'

        elif os.path.isdir(path):
            print 'This path is a directory. Keep copying ...'
            try:
                _, folderName = os.path.split(path)
                shutil.copytree(path,os.path.join(destinationFolder,folderName))
                print 'End of copying.'
                if isDelete:
                    print 'Deleting this directory ...'
                    try: shutil.rmtree(path); print 'End of deleting.\n'
                    except Exception as error: print 'Can not delete this directory.\nError message:\n'+str(error)+'\n'
                else: print ''
            except Exception as error: unCopied.append(path);print 'Can not copy this directory.\nError message:\n'+str(error)+'\n'
        else:
            unCopied.append(path)
            print 'This path is neither a file or a directory. Skip!\n'

    return unCopied


def importRawJCam(path,
                  dtype = np.dtype('>f'),
                  headerLength = 96, # length of the header, measured as the data type defined above
                  columnNumIndex = 14, # index of number of rows in header
                  rowNumIndex = 15, # index of number of columns in header
                  frameNumIndex = 16, # index of number of frames in header
                  decimation = None, #decimation number
                  exposureTimeIndex = 17): # index of exposure time in header, exposure time is measured in ms
    '''
    import raw JCam files into np.array


        raw file format:
        data type: 32 bit sigle precision floating point number
        data format: big-endian single-precision float, high-byte-first motorola
        header length: 96 floating point number
        column number index: 14
        row number index: 15
        frame number index: 16
        exposure time index: 17
    '''
    imageFile = np.fromfile(path,dtype=dtype,count=-1)

    columnNum = np.int(imageFile[columnNumIndex])
    rowNum = np.int(imageFile[rowNumIndex])

    if decimation is not None:
        columnNum /= decimation
        rowNum /= decimation

    frameNum = np.int(imageFile[frameNumIndex])

    if frameNum == 0: # if it is a single frame image
        frameNum += 1


    exposureTime = np.float(imageFile[exposureTimeIndex])

    imageFile = imageFile[headerLength:]

    print 'width =', str(columnNum), 'pixels'
    print 'height =', str(rowNum), 'pixels'
    print 'length =', str(frameNum), 'frame(s)'
    print 'exposure time =', str(exposureTime), 'ms'

    imageFile = imageFile.reshape((frameNum,rowNum,columnNum))

    return imageFile, exposureTime


def readBinaryFile(path,
                   position,
                   count = 1,
                   dtype = np.dtype('>f'),
                   whence = os.SEEK_SET):
    '''
    read arbitary part of a binary file,
    data type defined by dtype,
    start position defined by position (counts accordinating to dtype)
    length defined by count.
    '''

    f = open(path, 'rb')
    f.seek(position * dtype.alignment, whence)
    data = np.fromfile(f, dtype = dtype, count = count)
    f.close()
    return data


def readBinaryFile2(f,
                    position,
                    count = 1,
                    dtype = np.dtype('>f'),
                    whence = os.SEEK_SET):
    '''
    similar as readBinaryFile but without opening and closing file object
    '''
    f.seek((position * dtype.alignment), whence)
    data = np.fromfile(f, dtype = dtype, count = count)
    return data


def importRawJPhys(path,
                   dtype = np.dtype('>f'),
                   headerLength = 96, # length of the header for each channel
                   channels = ('photodiode2','read','trigger','photodiode'),# name of all channels
                   sf = 10000): # sampling rate, Hz
    '''
    import raw JPhys files into np.array
    one dictionary contains header for each channel
    the other contains values for each for each channel
    '''

    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError, 'Length of the file should be divisible by channel number!'

    header = {}
    body = {}

    for index, channelname in enumerate(channels):
        channelStart = index * channelLength
        channelEnd = channelStart + channelLength

        header.update({channels[index]: JPhysFile[channelStart:channelStart+headerLength]})
        body.update({channels[index]: JPhysFile[channelStart+headerLength:channelEnd]})

    body.update({'samplingRate':sf})

    return header, body


def importRawNewJPhys(path,
                      dtype = np.dtype('>f'),
                      headerLength = 96, # length of the header for each channel
                      channels = ('photodiode2',
                                  'read',
                                  'trigger',
                                  'photodiode',
                                  'sweep',
                                  'visualFrame',
                                  'runningRef',
                                  'runningSig',
                                  'reward',
                                  'licking'),# name of all channels
                      sf = 10000): # sampling rate, Hz
    '''
    import new style raw JPhys files into np.array
    one dictionary contains header for each channel
    the other contains values for each for each channel
    '''

    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum
#    print 'length of JPhys:', len(JPhysFile)
#    print 'length of JPhys channel number:', channelNum

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError, 'Length of the file should be divisible by channel number!'

    JPhysFile = JPhysFile.reshape([channelLength, channelNum])

    headerMatrix = JPhysFile[0:headerLength,:]
    bodyMatrix = JPhysFile[headerLength:,:]

    header = {}
    body = {}

    for index, channelname in enumerate(channels):

        header.update({channels[index]: headerMatrix[:,index]})
        body.update({channels[index]: bodyMatrix[:,index]})

    body.update({'samplingRate':sf})

    return header, body



def importRawJPhys2(path,
                    imageFrameNum,
                    photodiodeThr = .95, #threshold of photo diode signal,
                    dtype = np.dtype('>f'),
                    headerLength = 96, # length of the header for each channel
                    channels = ('photodiode2','read','trigger','photodiode'),# name of all channels
                    sf = 10000.): # sampling rate, Hz
    '''
    extract important information from JPhys file
    '''


    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if channelLength % 1 != 0:
        raise ArithmeticError, 'Bytes in each channel should be integer !'

    channelLength = int(channelLength)

    # get trace for each channel
    for index, channelname in enumerate(channels):
        channelStart = index * channelLength
        channelEnd = channelStart + channelLength
#        if channelname == 'expose':
#            expose = JPhysFile[channelStart+headerLength:channelEnd]

        if channelname == 'read':
            read = JPhysFile[channelStart+headerLength:channelEnd]

        if channelname == 'photodiode':
            photodiode = JPhysFile[channelStart+headerLength:channelEnd]

#        if channelname == 'trigger':
#            trigger = JPhysFile[channelStart+headerLength:channelEnd]

    # generate time stamp for each image frame
    imageFrameTS = []
    for i in range(1,len(read)):
        if read[i-1] < 3.0 and read[i] >= 3.0:
            imageFrameTS.append(i*(1./sf))

    if len(imageFrameTS) < imageFrameNum:
        raise LookupError, "Expose period number is smaller than image frame number!"
    imageFrameTS = imageFrameTS[0:imageFrameNum]

    # first time of visual stimulation
    visualStart = None

    for i in xrange(80,len(photodiode)):
        if ((photodiode[i] - photodiodeThr) * (photodiode[i-1] - photodiodeThr)) < 0 and \
           ((photodiode[i] - photodiodeThr) * (photodiode[i-75] - photodiodeThr)) < 0: #first frame of big change
                visualStart = i*(1./sf)
                break

    return np.array(imageFrameTS), visualStart


def importRawNewJPhys2(path,
                       imageFrameNum,
                       photodiodeThr = .95, #threshold of photo diode signal,
                       dtype = np.dtype('>f'),
                       headerLength = 96, # length of the header for each channel
                       channels = ('photodiode2',
                                   'read',
                                   'trigger',
                                   'photodiode',
                                   'sweep',
                                   'visualFrame',
                                   'runningRef',
                                   'runningSig',
                                   'reward',
                                   'licking'),# name of all channels
                       sf = 10000.): # sampling rate, Hz
    '''
    extract important information from new style JPhys file
    '''


    JPhysFile = np.fromfile(path,dtype=dtype,count=-1)
    channelNum = len(channels)

    channelLength = len(JPhysFile) / channelNum

    if len(JPhysFile) % channelNum != 0:
        raise ArithmeticError, 'Length of the file should be divisible by channel number!'

    JPhysFile = JPhysFile.reshape([channelLength, channelNum])

    bodyMatrix = JPhysFile[headerLength:,:]

    # get trace for each channel
    for index, channelname in enumerate(channels):

        if channelname == 'read':
            read = bodyMatrix[:,index]

        if channelname == 'photodiode':
            photodiode = bodyMatrix[:,index]

#        if channelname == 'trigger':
#            trigger = JPhysFile[channelStart+headerLength:channelEnd]

    # generate time stamp for each image frame
    imageFrameTS = []
    for i in range(1,len(read)):
        if (read[i-1] < 3.0) and (read[i] >= 3.0):
            imageFrameTS.append(i*(1./sf))

    if len(imageFrameTS) < imageFrameNum:
        raise LookupError, "Expose period number is smaller than image frame number!"
    imageFrameTS = imageFrameTS[0:imageFrameNum]

    # first time of visual stimulation
    visualStart = None

    for i in xrange(80,len(photodiode)):
        if ((photodiode[i] - photodiodeThr) * (photodiode[i-1] - photodiodeThr)) < 0 and \
           ((photodiode[i] - photodiodeThr) * (photodiode[i-75] - photodiodeThr)) < 0: #first frame of big change
                visualStart = i*(1./sf)
                break

    return np.array(imageFrameTS), visualStart


def getLog(logPath):
    '''
    get log dictionary from a specific path (including file names)
    '''

    f = open(logPath,'r')
    displayLog = pickle.load(f)
    f.close()
    return displayLog


def generateAVI(saveFolder,
                fileName,
                matrix,
                frameRate=25.,
                encoder='XVID',
                zoom=1,
                isDisplay=True
                ):

    '''
    :param saveFolder:
    :param fileName: can be with '.avi' or without '.avi'
    :param matrix: can be 3 dimensional (gray value) or 4 dimensional
                   if the length of the 4th dimension equals 3, it will be considered as rgb
                   if the length of the 4th dimension equals 4, it will be considered as rgba
    :param frameRate:
    :param encoder:
    :param zoom:
    :return: generate the .avi movie file
    '''

    if len(matrix.shape) == 4:
        if matrix.shape[3] == 3:
            r, g, b = np.rollaxis(matrix, axis = -1)
        elif matrix.shape[3] == 4:
            r, g, b, a = np.rollaxis(matrix, axis = -1)
        else: raise IndexError, 'The depth of matrix is not 3 or 4. Can not get RGB color!'
        r = r.reshape(r.shape[0],r.shape[1],r.shape[2],1)
        g = g.reshape(g.shape[0],g.shape[1],g.shape[2],1)
        b = b.reshape(b.shape[0],b.shape[1],b.shape[2],1)
        newMatix = np.concatenate((r,g,b),axis=3)
        newMatrix = (ia.array_nor(newMatix) * 255).astype(np.uint8)
    elif len(matrix.shape) == 3:
        s = (ia.array_nor(matrix) * 255).astype(np.uint8)
        s = s.reshape(s.shape[0],s.shape[1],s.shape[2],1)
        newMatrix = np.concatenate((s,s,s),axis=3)
    else: raise IndexError, 'The matrix dimension is neither 3 or 4. Can not get RGB color!'


    fourcc = cv2.cv.CV_FOURCC(*encoder)

    if fileName[-4:] != '.avi':
        fileName += '.avi'

    size = (int(newMatrix.shape[1]*zoom),int(newMatrix.shape[2]*zoom))

    filePath = os.path.join(saveFolder,fileName+'.avi')
    out = cv2.VideoWriter(filePath,fourcc, frameRate, size)

    for i in range(newMatrix.shape[0]):
        out.write(newMatrix[i,:,:,:])
        if isDisplay:
            cv2.imshow('movie',newMatrix[i,:,:,:])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


def importRawJCamF(path,
                   saveFolder = None,
                   dtype = np.dtype('<u2'),
                   headerLength = 116,
                   tailerLength = 218,
                   column = 2048,
                   row = 2048,
                   frame = None, #how many frame to read
                   crop = None):

    if frame:
        data = np.fromfile(path,dtype=dtype,count=frame*column*row+headerLength)
        header = data[0:headerLength]
        tailer = []
        mov = data[headerLength:].reshape((frame,column,row))
    else:
        data = np.fromfile(path,dtype=dtype)
        header = data[0:headerLength]
        tailer = data[len(data)-tailerLength:len(data)]
        frame = (len(data)-headerLength-tailerLength)/(column*row)
        mov = data[headerLength:len(data)-tailerLength].reshape((frame,column,row))

    if saveFolder:
        if crop:
            try:
                mov = mov[:,crop[0]:crop[1],crop[2]:crop[3]]
                fileName = path.split('\\')[-1] + '_cropped.tif'
            except Exception as e:
                print 'importRawJCamF: Can not understant the paramenter "crop":'+str(crop)+'\ncorp should be: [rowStart,rowEnd,colStart,colEnd]'
                print '\nTrace back: \n' + e
        else:
            fileName = path.split('\\')[-1] + '.tif'

        tf.imsave(os.path.join(saveFolder,fileName),mov)

    return mov, header, tailer


def int2str(num,length=None):
    '''
    generate a string representation for a integer with a given length
    :param num: input number
    :param length: length of the string
    :return: string represetation of the integer
    '''

    rawstr = str(int(num))
    if length is None or length == len(rawstr):return rawstr
    elif length < len(rawstr): raise ValueError, 'Length of the number is longer then defined display length!'
    elif length > len(rawstr): return '0'*(length-len(rawstr)) + rawstr


#==============================  obsolete  =========================================
#
# def getMatchingParameterDict(path):
#
#     with open(path,'r') as f:
#         txt = f.read()
#
#     chunkStart = txt.find('[VasculatureMapMatching]') + 25
#     chunkEnd = txt.find('[',chunkStart)
#     chunk = txt[chunkStart:chunkEnd]
#     paraTxtList = chunk.split('\n')
#
#     paraDict={}
#
#     for paraTxt in paraTxtList:
#         key, value = tuple(paraTxt.split(' = '))
#         if 'List' in key:
#             value = value.split(';')
#
#         if ('Hight' in key) or ('Width' in key) or ('Offset' in key):
#             value = int(value)
#         if (key == 'zoom') or (key == 'rotation'):
#             value = float(value)
#
#         paraDict.update({key:value})
#
#     return paraDict

#def importDeciJCamF(path,
#                    saveFolder = None,
#                    dtype = np.dtype('<u2'),
#                    headerLength = 0,
#                    tailerLength = 0,
#                    column = 2048,
#                    row = 2048,
#                    frame = None,
#                    crop = None):
#
#    if frame:
#        data = np.fromfile(path,dtype=dtype,count=frame*column*row+headerLength)
#        mov = data[headerLength:].reshape((frame,column,row))
#    else:
#        data = np.fromfile(path,dtype=dtype)
#        frame = (len(data)-headerLength-tailerLength)/(column*row)
#        mov = data[headerLength:len(data)-tailerLength].reshape((frame,column,row))
#
#    if saveFolder:
#
#        if crop.any():
#            mov = mov[:,crop[0]:crop[1],crop[2]:crop[3]]
#            fileName = path.split('\\')[-1] + '_cropped.tif'
#        else:
#            fileName = path.split('\\')[-1] + '.tif'
#
#        tf.imsave(os.path.join(saveFolder,fileName),mov)
#
#    return mov


if __name__=='__main__':

    #----------------------------------------------------------------------------
    print int2str(5)
    print int2str(5,2)
    print int2str(155,6)
    #----------------------------------------------------------------------------


    print 'well done!'

