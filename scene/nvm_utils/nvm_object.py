# NVM Object

class NvmObject:
    def __init__(self):
        # variables for nvm manipulation
        self.nvmVersion = ""
        self.nvmCalibration = ""
        self.numCamerasTotal = 0
        self.numPointsTotal = 0
        self.modelArray = []
        self.numFullModels = 0
        self.numEmptyModels = 0
        self.numTotalModels = 0
        self.plyArray = []
        self.numPlyFiles = 0


class ModelObject:
    def __init__(self):
        self.numCameras = 0
        self.cameraArray = []  # array of CameraObject s
        self.numPoints = 0
        self.pointArray = []  # array of PointObject s


class CameraObject:
    def __init__(self):
        self.fileName = ""
        self.focalLength = ""
        self.quaternionArray = ["", "", "", ""]
        self.cameraCenter = ["", "", ""]
        self.radialDistortion = ""


class PointObject:
    def __init__(self):
        self.xyzArray = ["", "", ""]
        self.rgbArray = ["", "", ""]
        self.numMeasurements = 0
        self.measurementArray = []  # array of PointMeasurementObject s


class PointMeasurementObject:
    def __init__(self):
        self.imageIndex = ""
        self.featureIndex = ""
        self.xyArray = ["", ""]
