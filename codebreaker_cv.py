import re
import cv2
import os
import sys
import numpy as np
import operator
import boto3 as aws


class AWS:
    @staticmethod
    def initConfig():
        os.environ['AWS_SHARED_CREDENTIALS_FILE'] = 'resources/aws/credentials.ini'
        os.environ['AWS_CONFIG_FILE '] = 'resources/aws/config.ini'


class CharacterRecognitionWithRekognition:
    def __init__(self):
        # initialize AWS configuration and credentials
        AWS.initConfig()
        self.service = "rekognition"
        self.region = "us-west-2"
        self.rekognition = aws.client(self.service, self.region)

    @staticmethod
    def __cleanAlphabetsIfPresent(text):
        """
        :param: string
        :return: string without any non alphabetic characters if the string had aplhabets earlier
        """
        text = re.sub('[^a-zA-Z0-9]', '', text)
        if re.search('[a-zA-Z]', text):
            return re.sub('[^a-zA-Z]', '', text.upper())
        else:
            return text

    def imageToText(self, image):
        """
        :param: image in bytes
        :return: list of detected text from the image
        """
        response = self.rekognition.detect_text(Image={'Bytes': image})
        detectedText = set()
        for x in response['TextDetections']:
            text = self.__cleanAlphabetsIfPresent(x['DetectedText'])
        if not len(text) == 0:
            detectedText.add(text)
        return list(detectedText)


class _ContourWithData:
    """
    Inspired from [https://github.com/MicrocontrollersAndMore/OpenCV_3_KNN_Character_Recognition_Python]
    """

    def __init__(self, minimumContourArea=100):
        self.npaContour = None  # contour
        self.boundingRect = None  # bounding rect for contour
        self.intRectX = 0  # bounding rect top left corner x location
        self.intRectY = 0  # bounding rect top left corner y location
        self.intRectWidth = 0  # bounding rect width
        self.intRectHeight = 0  # bounding rect height
        self.fltArea = 0.0  # area of contour
        self.minimumContourArea = minimumContourArea

    def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):  # this is oversimplified, for a production grade program
        if self.fltArea < self.minimumContourArea:
            return False  # much better validity checking would be necessary
        else:
            return True


class CharacterRecognitionWithKNN:
    """
    Inspired from [https://github.com/MicrocontrollersAndMore/OpenCV_3_KNN_Character_Recognition_Python]
    """

    def __init__(self, minimumContourArea=100, resizedImageWidth=20,
                 resizedImageHeight=30):
        self.minimumContourArea = minimumContourArea
        self.resizedImageWidth = resizedImageWidth
        self.resizedImageHeight = resizedImageHeight

        self.numbersTrainingImage = "generated/ocr/trainingNumbers.png"
        self.alphabetsTrainingImage = "generated/ocr/trainingAlphabets.png"
        self.numbersClassificationsFile = "generated/ocr/numbersClassifications.txt"
        self.alphabetsClassificationsFile = "generated/ocr/alphabetsClassifications.txt"
        self.numbersFlattenedImagesFile = "generated/ocr/numbersFlattenedImages.txt"
        self.alphabetsFlattenedImagesFile = "generated/ocr/alphabetsFlattenedImages.txt"

        self.kNearestNumbers, self.kNearestAlphabets = self.trainModelsForCharacterRecognitionUsingKNN()  # train and get the KNN model

    def generateDataForCharacterRecognitionUsingKNN(self):
        """
        :param: None
        :return: None, it generates data as .txt files for further OCR KNN model training
        """
        if self.__trainWithImage(0, self.numbersTrainingImage, self.numbersClassificationsFile,
                                 self.numbersFlattenedImagesFile) and self.__trainWithImage(1,
                                                                                            self.alphabetsTrainingImage,
                                                                                            self.alphabetsClassificationsFile,
                                                                                            self.alphabetsFlattenedImagesFile):
            print("Training data generated.")
        else:
            print("Error occurred while generating training data.")

    def __trainWithImage(self, mode, trainingImage, classificationsFile, flattenedImagesFile):
        """
        :param: mode=[1 for alphabets, 0 for numbers], trainingImage=[path to file], classificationsFile=[path to file],
                flattenedImagesFile=[path to file]
        :return: None, it generates data as .txt files for further OCR KNN model training
        """
        imgTraining = cv2.imread(trainingImage)  # read in training numbers image

        if imgTraining is None:  # if image was not read successfully
            print("Error: Image not read from file.")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return False  # and exit function (which exits program)
        # end if

        # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
        if mode == 0:
            intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'),
                             ord('8'),
                             ord('9')]
        elif mode == 1:
            intValidChars = [ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'),
                             ord('I'),
                             ord('J'), ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'),
                             ord('R'),
                             ord('S'), ord('T'), ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
        else:
            print("Error: Invalid mode.")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return False  # and exit function (which exits program)

        imgGray = cv2.cvtColor(imgTraining, cv2.COLOR_BGR2GRAY)  # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

        # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                          255,  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,
                                          # invert so foreground will be white, background will be black
                                          11,  # size of a pixel neighborhood used to calculate threshold value
                                          2)  # constant subtracted from the mean or weighted mean

        cv2.imshow("imgThresh", imgThresh)  # show threshold image for reference

        imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                                  # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                                  cv2.RETR_EXTERNAL,
                                                                  # retrieve the outermost contours only
                                                                  cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

        # declare empty numpy array, we will use this to write to file later
        # zero rows, enough cols to hold all image data
        npaFlattenedImages = np.empty((0, self.resizedImageWidth * self.resizedImageHeight))

        intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

        for npaContour in npaContours:  # for each contour
            if cv2.contourArea(npaContour) > self.minimumContourArea:  # if contour is big enough to consider
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

                # draw rectangle around each contour as we ask user for input
                cv2.rectangle(imgTraining,  # draw rectangle on original training image
                              (intX, intY),  # upper left corner
                              (intX + intW, intY + intH),  # lower right corner
                              (0, 0, 255),  # red
                              2)  # thickness

                imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
                imgROIResized = cv2.resize(imgROI, (self.resizedImageWidth,
                                                    self.resizedImageHeight))  # resize image, this will be more consistent for recognition and storage

                cv2.imshow("imgROI", imgROI)  # show cropped out char for reference
                cv2.imshow("imgROIResized", imgROIResized)  # show resized image for reference
                cv2.imshow("trainingImage.png",
                           imgTraining)  # show training image, this will now have red rectangles drawn on it

                intChar = cv2.waitKey(0)  # get key press

                if intChar == 27:  # if esc key was pressed
                    sys.exit()  # exit program
                elif intChar in intValidChars:  # else if the char is in the list of chars we are looking for . . .

                    intClassifications.append(
                        intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)

                    npaFlattenedImage = imgROIResized.reshape((1,
                                                               self.resizedImageWidth * self.resizedImageHeight))  # flatten image to 1d numpy array so we can write to file later
                    npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                                   0)  # add current flattened image numpy array to list of flattened image numpy arrays
                else:
                    print("Invalid key press. Use only numbers and capital letters. Training corrupted. Exiting")
                    sys.exit()  # exit program
                # end if
            # end if
        # end for

        fltClassifications = np.array(intClassifications,
                                      np.float32)  # convert classifications list of ints to numpy array of floats

        npaClassifications = fltClassifications.reshape(
            (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

        # print("Training data generated.")
        # print(npaFlattenedImages)
        # print(npaClassifications)
        np.savetxt(classificationsFile, npaClassifications)
        np.savetxt(flattenedImagesFile, npaFlattenedImages)  # write flattened images to file

        cv2.destroyAllWindows()  # remove windows from memory

        return True

    def trainModelsForCharacterRecognitionUsingKNN(self):
        """
        :param: None
        :return: Trained KNN model object
        """
        return self.__trainModel(self.numbersClassificationsFile, self.numbersFlattenedImagesFile), self.__trainModel(
            self.alphabetsClassificationsFile, self.alphabetsFlattenedImagesFile)

    @staticmethod
    def __trainModel(classificationsFile, flattenedImagesFile):
        """
        :param: classificationsFile=[path to file], flattenedImagesFile=[path to file]
        :return: Trained KNN model object
        """
        try:
            npaClassifications = np.loadtxt(classificationsFile, np.float32)  # read in training classifications
        except:
            print("Error: Unable to open " + classificationsFile + ", exiting program.")
            os.system("pause")
            return
        # end try

        try:
            npaFlattenedImages = np.loadtxt(flattenedImagesFile, np.float32)  # read in training images
        except:
            print("Error: Unable to open " + flattenedImagesFile + ", exiting program.")
            os.system("pause")
            return
        # end try

        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
        kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return kNearest

    def detectNumbers(self, image):
        """
        :param: cv2 image
        :return: string of detected numbers from the image
        """
        return self.__imageToText(image, self.kNearestNumbers)

    def detectAlphabets(self, image):
        """
        :param: cv2 image
        :return: string of detected alphabets from the image
        """
        return self.__imageToText(image, self.kNearestAlphabets)

    def __imageToText(self, image, knnModel):
        """
        :param: cv2 image, trained KNN model
        :return: list of detected text from the image
        """
        allContoursWithData = []  # declare empty lists,
        validContoursWithData = []  # we will fill these shortly

        kNearest = knnModel

        # imgTestingNumbers = cv2.imread(image)  # read in testing numbers image
        imgTestingNumbers = image

        if imgTestingNumbers is None:  # if image was not read successfully
            print("Error: Image not read from file.")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return None  # and exit function (which exits program)
        # end if

        imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

        # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                          255,  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,
                                          # invert so foreground will be white, background will be black
                                          11,  # size of a pixel neighborhood used to calculate threshold value
                                          2)  # constant subtracted from the mean or weighted mean

        imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                                  # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                                  cv2.RETR_EXTERNAL,
                                                                  # retrieve the outermost contours only
                                                                  cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

        for npaContour in npaContours:  # for each contour
            contourWithData = _ContourWithData(
                minimumContourArea=self.minimumContourArea)  # instantiate a contour with data object
            contourWithData.npaContour = npaContour  # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
            allContoursWithData.append(
                contourWithData)  # add contour with data object to list of all contours with data
        # end for

        for contourWithData in allContoursWithData:  # for all contours
            if contourWithData.checkIfContourIsValid():  # check if valid
                validContoursWithData.append(contourWithData)  # if so, append to valid contour list
            # end if
        # end for

        validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right

        strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program

        for contourWithData in validContoursWithData:  # for each contour
            # draw a green rect around the current char
            # cv2.rectangle(imgTestingNumbers,  # draw rectangle on original testing image
            #               (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
            #               (contourWithData.intRectX + contourWithData.intRectWidth,
            #                contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
            #               (0, 255, 0),  # green
            #               2)  # thickness

            imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                     # crop char out of threshold image
                     contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (self.resizedImageWidth,
                                                self.resizedImageHeight))  # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape(
                (1, self.resizedImageWidth * self.resizedImageHeight))  # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                         k=1)  # call KNN function find_nearest

            strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

            strFinalString = strFinalString + strCurrentChar  # append current char to full string
        # end for

        # print(strFinalString)  # show the full string
        #
        # cv2.imshow("imgTestingNumbers",
        #            imgTestingNumbers)  # show input image with green boxes drawn around found digits
        # cv2.waitKey(0)  # wait for user key press
        #
        # cv2.destroyAllWindows()  # remove windows from memory

        return strFinalString


class PuzzleDetection:

    @staticmethod
    def __findGrid(image):
        """
        :param image: input image containing puzzle in opencv format
        :return: bounding box (top-left-x, top-left-y, width, height) and the area (a) for the puzzle grid
        """
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageGrayBlurred = cv2.GaussianBlur(imageGray, (5, 5), 0)
        imageGrayBlurredThreshold = cv2.adaptiveThreshold(imageGrayBlurred, 255, 1, 1, 11, 2)

        # display image
        # cv2.imshow('TEST', cv2.resize(imageGrayBlurredThreshold, (600, 600)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, contours, hierarchy = cv2.findContours(imageGrayBlurredThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggestSquare = None
        maxSquareArea = 0
        minSquareArea = imageGrayBlurredThreshold.size / 4

        for contour in contours:
            squareArea = cv2.contourArea(contour)
            if squareArea > minSquareArea:
                perimeter = cv2.arcLength(contour, True)
                approxOutline = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if squareArea > maxSquareArea and len(approxOutline) == 4:
                    biggestSquare = approxOutline
                    maxSquareArea = squareArea

        x = biggestSquare[0][0][0]
        y = biggestSquare[0][0][1]
        w = biggestSquare[2][0][0] - biggestSquare[0][0][0]
        h = biggestSquare[2][0][1] - biggestSquare[0][0][1]
        a = maxSquareArea

        # Perspective Transformation
        # pts1 = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        # pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
        # M = cv2.getPerspectiveTransform(pts1, pts2)
        # dst = cv2.warpPerspective(image, M, (600, 600))
        #
        # # display image
        # cv2.imshow('DST', dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return x, y, w, h, a

    @staticmethod
    def __groupAndSortGridPoints(points, radius=5, method=0):
        """
        :param points: a list of points to be grouped and sorted
        :param radius: difference between points to group
        :param method: group and sort row wise (0) or column wise(1).
        :return: a 2D list of points with grouped and sorted points based on radius and method.
        """
        t = 1
        f = 0
        if method == 1:
            t = 0
            f = 1
        else:
            t = 1
            f = 0
        allPoints = set()
        for i in range(len(points)):
            group = list()
            for j in range(len(points)):
                difference = abs(points[i][t] - points[j][t])
                if difference <= radius:
                    group.append(points[j])
            allPoints.add(tuple(sorted(group, key=lambda x: x[f])))
        return sorted([list(x) for x in allPoints], key=lambda x: x[0][t])

    @staticmethod
    def __averageNearestPoints(points, radius=10):
        """
        :param radius: difference between points to average
        :param points: a list of points to be averaged
        :return: a list of points with nearest points based on radius averaged into one.
        """
        allPoints = set()
        for i in range(len(points)):
            neighbors = list()
            for j in range(len(points)):
                difference = (abs(points[i][0] - points[j][0]), abs(points[i][1] - points[j][1]))
                if max(difference[0], difference[1]) <= radius:
                    neighbors.append(points[j])
            allPoints.add((round(sum([n[0] for n in neighbors]) / len(neighbors)),
                           round(sum([n[1] for n in neighbors]) / len(neighbors))))
        return list(allPoints)

    def __extractGridPoints(self, image, area, puzzleSize):
        """
        :param image: input image (cropped bounding box of a puzzle) containing grid in opencv format
        :param area: area of the input image (cropped bounding box of a puzzle)
        :param puzzleSize: size of the puzzle grid (as number of squares)
        :return: a list of grid points
        """
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageGrayBlurred = cv2.GaussianBlur(imageGray, (5, 5), 0)
        imageGrayBlurredThreshold = cv2.adaptiveThreshold(imageGrayBlurred, 255, 1, 1, 11, 2)

        _, contours, hierarchy = cv2.findContours(imageGrayBlurredThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maxSquareArea = (area / (puzzleSize * puzzleSize)) * 1.5
        minSquareArea = (area / (puzzleSize * puzzleSize)) * 0.5
        gridPoints = list()
        # add initial 4 corners to the points list
        gridPoints.append((1, 1))
        gridPoints.append((1, 1 + (area ** 0.5)))
        gridPoints.append((1 + (area ** 0.5), 1))
        gridPoints.append((1 + (area ** 0.5), 1 + (area ** 0.5)))
        for contour in contours:
            squareArea = cv2.contourArea(contour)
            if minSquareArea < squareArea < maxSquareArea:
                x, y, w, h = cv2.boundingRect(contour)
                gridPoints.append((x, y))
                gridPoints.append((x + w, y))
                gridPoints.append((x + w, y + h))
                gridPoints.append((x, y + h))
        return self.__averageNearestPoints(gridPoints)

    @staticmethod
    def __extractGridBoxes(image, points):
        """
        :param image: input image (cropped bounding box of a puzzle) containing grid in opencv format
        :param points: a list of the grid points
        :return: a 2D list containing each puzzle box
        """
        borderError = 4
        edgeLength = len(points) - 1
        grid = list()
        for i in range(edgeLength):
            row = list()
            for j in range(edgeLength):
                row.append(image[points[i][j][1] + borderError: points[i + 1][j][1] - borderError,
                           points[i][j][0] + borderError: points[i][j + 1][0] - borderError])
            grid.append(row)
        return grid

    def __fillGridBoxes(self, image, points, data):
        """
        :param image: input image (cropped bounding box of a puzzle) containing grid in opencv format
        :param points: a list of the grid points
        :param data: data to filled into the image
        :return: the puzzle image with filled data
        """
        # text style
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 3
        fontColor = (0, 0, 255)
        lineThickness = 8
        lineType = 2

        borderError = 4

        # draw the text
        edgeLength = len(points) - 1
        for i in range(edgeLength):
            for j in range(edgeLength):
                if self.__imageIsWhite(image[points[i][j][1] + borderError: points[i + 1][j][1] - borderError,
                                       points[i][j][0] + borderError: points[i][j + 1][0] - borderError]):
                    x = points[i][j][0]
                    w = points[i][j + 1][0] - points[i][j][0]
                    y = points[i][j][1]
                    h = points[i + 1][j][1] - points[i][j][1]
                    textPosition = ((x + int((w / 2))) - (fontScale * 10), (y + int((h / 2))) + (fontScale * 10))
                    cv2.putText(image, str(data[i][j]), textPosition, font, fontScale, fontColor, lineThickness,
                                lineType)
        return image

    @staticmethod
    def __imageIsBlack(image):
        """
        :param: image in opencv format
        :return: True if image has >=95% black pixels, otherwise False
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        if (cv2.countNonZero(image) / (h * w)) * 100 < 5:
            return True
        else:
            # print((cv2.countNonZero(image) / (h * w)) * 100)
            return False

    @staticmethod
    def __imageIsWhite(image):
        """
        :param: image in opencv format
        :return: True if image has >98% white pixels, otherwise False
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        if (np.sum(image == 255) / (h * w)) * 100 > 98:
            return True
        else:
            # print((cv2.countNonZero(image) / (h * w)) * 100)
            return False

    @staticmethod
    def __convertImageFormatFromBufferToOpenCV(imageBuffer):
        """
        :param: image as buffer
        :return: image in opencv format
        """
        return cv2.imdecode(imageBuffer, cv2.IMREAD_COLOR)

    @staticmethod
    def __convertImageFormatFromOpenCVToBuffer(image):
        """
        :param: image in opencv format
        :return: image as buffer
        """
        extension = 'PNG'
        return cv2.imencode(extension, image)

    def detectSudokuPuzzle(self, image, puzzleSize):
        """
        :param image: input image as buffer
        :param puzzleSize: size of the puzzle grid (as number of squares)
        :return: a 2D matrix of the puzzle grid
        """
        return True, [[]]

    def detectCodeWordPuzzle(self, image, puzzleSize):
        """
        :param image: input image as buffer
        :param puzzleSize: size of the puzzle grid (as number of squares)
        :return: a 2D matrix of the puzzle grid, a dict that maps 1-26
        """
        try:
            # convert image buffer to opencv format
            # image = self.__convertImageFormatFromBufferToOpenCV(image)

            # extract the squares from the puzzle grid
            x, y, w, h, a = self.__findGrid(image)
            puzzleSquare = image[y: y + h, x: x + w]
            gridPoints = self.__groupAndSortGridPoints(self.__extractGridPoints(puzzleSquare, a, puzzleSize))
            # print(len(gridPoints) ** 2)
            # for group in gridPoints:
            #     for point in group:
            #         cv2.circle(puzzleSquare, point, 6, (0, 0, 255), -1)
            # print(point)
            # cv2.imshow('Result', cv2.resize(puzzleSquare, (600, 600)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            grid = self.__extractGridBoxes(puzzleSquare, gridPoints)

            # pass each detected square through OCR and create a digital representation
            rekognitionOCR = CharacterRecognitionWithRekognition()
            knnOCR = CharacterRecognitionWithKNN(minimumContourArea=55)
            data = [[0] * (len(gridPoints) - 1) for _ in range(len(gridPoints) - 1)]

            # only required for codeword puzzle
            table = dict.fromkeys([num for num in range(1, 27)])

            for i in range(len(grid)):
                for j in range(len(grid)):
                    # for codeword puzzle
                    if not self.__imageIsBlack(grid[i][j]):
                        # detect and fill data using KNN
                        h, w = grid[i][j].shape[:2]
                        upperHalf = grid[i][j][:int(h / 2.5), :]
                        lowerHalf = grid[i][j][int(h / 2.5):, :]
                        number = knnOCR.detectNumbers(upperHalf)
                        if number is not None:
                            if not len(number) == 0:
                                data[i][j] = int(number)
                        if not self.__imageIsWhite(lowerHalf):
                            alphabet = knnOCR.detectAlphabets(lowerHalf)
                            if alphabet is not None:
                                if not len(alphabet) == 0:
                                    table[int(number)] = alphabet

                        # detect and fill data using AWS Rekognition
                        # imagePath = 'images/box' + str(i) + '-' + str(j) + '.jpg'
                        # cv2.imwrite(imagePath, grid[i][j])
                        # cv2.imwrite(imagePath, cv2.resize(grid[i][j], (320, 320)))
                        # with open(imagePath, "rb") as imageFile:
                        #     detectedText = rekognitionOCR.imageToText(bytearray(imageFile.read()))
                        #     hasAlpha = False
                        #     key = int()
                        #     value = str()
                        #     for d in detectedText:
                        #         if d.isalpha():
                        #             hasAlpha = True
                        #             value = d
                        #         else:
                        #             key = int(d)
                        #     data[i][j] = key
                        #     if hasAlpha:
                        #         table[key] = value
                        # os.remove(imagePath)

                        # cv2.imshow('Puzzle Box ' + str(i) + ', ' + str(j), grid[i][j])
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
            return True, data, table
        except:
            return False, [[]], dict()
