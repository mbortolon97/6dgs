# #outline of .nvm file

# NVM_V3 [optional calibration]                                         #file version header
# <Number of cameras>
# <File Name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# .
# .
# .
# <File Name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# [optional blank line]
# <Number of 3D points>
# <XYZ> <RBG> <number of measurements> <Image index> <Feature Index> <xy> ... <Image index> < Feature Index> <xy>

# .
# .
# .

# <Number of cameras>
# <File Name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# .
# .
# .
# <File Name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# [optional blank line]
# <Number of 3D points>
# <XYZ> <RBG> <number of measurements> <Image index> <Feature Index> <xy> ... <Image index> < Feature Index> <xy>
# [optional blank line]
# 0
# <PLY comments>
# <number of PLY files>
# <List of indices of models that have associated PLY>
# #

##########################################################

# nvmObject, which contains an array of models, the number of full models, the number of empty models, the total
# number of models, an array of PLY files, and the number of PLY files

import sys
from nvmObject import *


# read through any blank or commented lines
def skipBlankLines(f):
    line = ""
    while True:
        line = f.readline()
        line = line.rstrip()
        if (not (line.find('#') == -1)):
            line = line[0:line.find('#')]
        if (not len(line) == 0):
            break
    return line


# functions for nvm manipulation
def readNvm(inputFile):
    nvmObject = NvmObject()
    # extracting and parsing info from nvm file
    with open(inputFile) as f:
        readVersion(f, nvmObject)  # parsing the NVM version and configuration
        readModels(f, nvmObject)  # parsing the Models from the NVM
        # readPLY(f, nvmObject) # parsing the PLY files from the NVM
    return nvmObject


def readVersion(f, nvmObject):
    line = skipBlankLines(f)
    # read in version (don't know how version will change input yet)
    if (not (line.find(' ') == -1)):  # there is more than one word
        nvmObject.nvmVersion = line[0:line.find(' ')]  # getting just the version, no calibration
        # calibration command in file read 'FixedK fx cx fy cy'
        nvmObject.nvmCalibration = line[len(nvmObject.nvmVersion):]
    else:  # there is only one word
        nvmObject.nvmVersion = line


def readModels(f, nvmObject):
    # nvmObject has nvmVersion, nvmCalibration, numCamerasTotal, numPointsTotal, modelArray,
    # numFullModels, numEmptyModels, numTotalModels, plyArray, numPlyFiles

    # loop through models
    while True:
        line = skipBlankLines(f)  # read through any blank or comment lines

        if line[0] == '0': break  # stop reading models if input is just "0"

        # gather model data
        modelObject = ModelObject()  # has numCameras, cameraArray, numPoints, pointArray
        nvmObject.modelArray.append(modelObject)
        nvmObject.numTotalModels += 1
        modelObject.numCameras = int(line[0:])  # read in number of cameras
        nvmObject.numCamerasTotal += modelObject.numCameras
        readCameras(f, modelObject)  # read in list of cameras

        line = skipBlankLines(f)
        modelObject.numPoints = int(line[0:])  # read in number of 3D points
        nvmObject.numPointsTotal += modelObject.numPoints
        if modelObject.numPoints > 0:
            nvmObject.numFullModels += 1
        else:
            nvmObject.numEmptyModels += 1
        readPoints(f, modelObject)  # read in 3D point attributes
    # end of while reading through all models


def readCameras(f, modelObject):
    # modelObject has numCameras, cameraArray, numPoints, pointArray
    x = 0
    while x < modelObject.numCameras:  # reading in however many cameras are in this model
        cameraObj = CameraObject()  # has fileName, focalLength, quaternionArray, cameraCenter, radialDistortion
        modelObject.cameraArray.append(cameraObj)
        line = skipBlankLines(f)

        # read in file name
        cameraObj.fileName = line[
                             0:line.find('        ')]  # get each camera file location and store it #strange character
        line = line[line.find('        ') + 1:]  # removing filename from temp reading line #strange character
        line = line.strip()
        # read in focal length
        cameraObj.focalLength = line[0:line.find(' ')]  # <focal length> --> one integer
        line = line[line.find(' ') + 1:]  # removing focal length from temp reading line
        # read in quaternion <WXYZ>
        y = 0
        while y < 4:
            cameraObj.quaternionArray[y] = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            y += 1
            # read in camera center <XYZ>
        y = 0
        while y < 3:
            cameraObj.cameraCenter[y] = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            y += 1
            # read in radial distortion
        cameraObj.radialDistortion = line[0:line.find(' ')]  # <radial distortion> --> one int
        # there is a zero after each camera, so don't worry about the rest of the line
        x += 1
    # end of while x


def readPoints(f, modelObject):
    # modelObject has numCameras, cameraArray, numPoints, pointArray
    x = 0
    while x < modelObject.numPoints:  # reading in however many cameras are in this model
        pointObj = PointObject()  # has xyzArray, rgbArray, numMeasurments, measurementArray[]
        modelObject.pointArray.append(pointObj)
        line = skipBlankLines(f)
        # read in <XYZ>
        y = 0
        while y < 3:
            pointObj.xyzArray[y] = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            line = line.strip()
            y += 1
        # read in <RGB>
        y = 0
        while y < 3:
            pointObj.rgbArray[y] = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            line = line.strip()
            y += 1
        # read in number of measurements
        pointObj.numMeasurements = int(line[0:line.find(' ')])
        line = line[line.find(' ') + 1:]
        # read in list of measurements
        y = 0
        while y < pointObj.numMeasurements:
            measObj = PointMeasurementObject()  # has imageIndex, featureIndex, xyArray[]
            pointObj.measurementArray.append(measObj)
            # read in image index
            measObj.imageIndex = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            line = line.strip()
            # read in feature index
            measObj.featureIndex = line[0:line.find(' ')]
            line = line[line.find(' ') + 1:]
            line = line.strip()
            # read in <XY>
            z = 0
            while z < 2:
                # this if-else is to handle reading information from the very end of a line
                if y < pointObj.numMeasurements - 1:  # NOT reading in the last measurment
                    measObj.xyArray[z] = line[0:line.find(' ')]  # not the last number in the line
                else:  # yes reading in the last measurement
                    if z == 0:
                        measObj.xyArray[z] = line[0:line.find(' ')]
                    else:
                        measObj.xyArray[z] = line[0:]  # yes the last number in the line
                line = line[line.find(' ') + 1:]
                line = line.strip()
                z += 1
            # end of while reading through <xy>
            y += 1
        # end of while reading through measurements
        x += 1
    # end of while reading through points

# def readPLY(f, nvmObject):
# read in int for number of PLY files
# read in list of indices of models that have associated PLY