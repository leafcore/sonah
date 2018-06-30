"""This is an example module showing how the API should be used."""
from api.hackathon import HackathonApi, RunModes
import os
import time
import cv2
import matplotlib.pyplot as plt
from tesserwrap import Tesseract
from PIL import Image
import numpy as np
import itertools

from helper import four_point_transform

class MySolution(HackathonApi):
    """
    This class implements your solution to the two tasks.

    When running the agorithms via the HackathonAPI's run method,
    the methods below are automatically called when needed.
    However you can also execute them yourself, by directly invoking them with
    the necessary parameters.

    The HackathonApi class does not implement a __init__ constructor,
    therefore you can implement your own here and don't have to care about
    matching any super-class method signature.
    Instead take care to call the initializeApi method of this class' object,
    if you want to use the Hackathon API. (See below for an example)

    You can also create other files and classes and do whatever you like, as
    long as the methods below return the required values.
    """

    def handleFrameForTaskA(self, frame):
        """
        See the documentation in the parent class for a whole lot of information on this method.

        We will just stupidly return random ROIs here
        to show what the result has to look like.
        """
        resultSet = [[
            np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
            np.array([[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]),
            np.array([[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]),
            np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]])
        ], [
            np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 0.35], [0.5, 0.35]])
        ], [
            np.array([[0.2, 0.3], [0.5, 0.3], [0.5, 0.35], [0.2, 0.35]])
        ], [
            np.array([[0.3, 0.6], [0.5, 0.6], [0.5, 0.65], [0.3, 0.65]]),
            np.array([[0.1, 0.2], [0.3, 0.25], [0.3, 0.3], [0.1, 0.27]]),
        ], [
            np.array([[0.8, 0.5], [0.9, 0.5], [0.9, 0.52], [0.8, 0.52]])
        ]]
        return resultSet[int((time.time() / 10.0) % len(resultSet))]

    def handleFrameForTaskB(self, frame, regionCoordinates):
        try:
            coordinates = list()
            for point in regionCoordinates:
                coordinates.append([point[0]*frame.shape[1], point[1]*frame.shape[0]])
            coordinates = np.int0(coordinates)
            frame = cv2.drawContours(frame, [coordinates], 0, (0, 255, 0), 2)
            warped = four_point_transform(frame, coordinates)
            shrunk = cv2.cvtColor(warped[:, int(warped.shape[1]/10):], cv2.COLOR_BGR2GRAY)
            scale = 6
            shrunk = cv2.resize(shrunk, (shrunk.shape[1]*scale,shrunk.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
            _, shrunk = cv2.threshold(shrunk, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            shrunk = 255-cv2.dilate(255-shrunk, np.ones((1, 1)), iterations=1)

            num, features = cv2.connectedComponents(255-shrunk)

            plate = str()
            corners = list()
            for i in range(0, num):
                area = np.sum((features==i))
                if area > scale**2*2*25 and area < scale*4*500:
                    rows = np.any(features == i, axis=1)
                    cols = np.any(features == i, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    corners.append([rmin, cmin, rmax, cmax])
            corners = np.array(corners)

            idx = np.argsort(corners[:,1])
            sorted_corners = corners[idx]

            for corner in sorted_corners:
                minx = corner[0]-2
                miny = corner[1]-2
                maxx = corner[2]+2
                maxy = corner[3]+2

                if minx < 0:
                    minx = 0
                if miny < 0:
                    miny = 0

                snip = features[minx:maxx, miny:maxy]

                if snip.shape[1] > snip.shape[0]:
                    continue

                snip = cv2.erode(snip.astype(np.uint8), np.ones((5,5)), iterations=1)
                im = Image.fromarray(np.uint8(snip))
                tr = Tesseract(datadir="/usr/share/tessdata")

                letter = tr.ocr_image(im).rstrip()
                for l in letter:
                    if l.isalnum():
                        letter = l
                plate += letter.capitalize()

            alphs = "".join(itertools.takewhile(str.isalpha, plate))
            nums = plate[len(alphs):]

            if len(alphs) == 2:
                plate = alphs[0] + "-" + alphs[1] + "-" + nums
            elif len(alphs) == 5:
                plate = alphs[:3] + "-" + alphs[3:] + "-" + nums
            else:
                diffs = list()
                alphscorners = sorted_corners[:len(alphs)]
                for i in range(len(alphscorners)):
                    if sorted_corners[i][1] == alphscorners[-1][1]:
                        break
                    diffs.append(sorted_corners[i+1][1]-sorted_corners[i][3])

                cuts = np.array(diffs) > np.mean(diffs)
                rev_cuts = cuts[::-1]
                for i in range(len(cuts[::-1])):
                    if (rev_cuts[i] == 1):
                        alphs = alphs[:len(cuts)-i] + "-" + alphs[len(cuts)-i:]
                plate = alphs + "-" + nums
            if len(plate) < 5:
                return None
            elif len(plate) > 11:
                return None
            elif plate.count("-") > 2:
                return None
            elif plate.count("-") < 2:
                return None
            else:
                return plate
        except Exception as exception:
            return None


if __name__ == "__main__":
    """This is an example of how to use the Hackathon API."""
    # We instantiate an object of our implemented solution first.
    solution = MySolution()
    # Before running the code, the hackathon API has to be initialized.
    # This loads the metadata, needed for running things automatically.
    # Make sure you downloaded all the frames with the download_labeldata.sh script.
    datasetWrapper = solution.initializeApi(os.path.abspath("./metadata.json"), os.path.abspath("./data/"))
    print("MySolution begins here...")
    print("The total number of frames is {:d}".format(datasetWrapper.getTotalFrameCount()))
    # We can test our implementation in multiple modes, by supplying a RunMode
    # to the run method. This will automatically print a few stats after running.
    # You may however implement your own stats, by using the methods
    # of the datasetWrapper directly. You can get frames with its
    # getFrame(frameId) method for example. Have a look at the class' documentation
    # inside the ./api/hackathon.py file!
    solution.run(RunModes.TASK_B_FULL)
    solution.run(RunModes.INTEGRATED_FULL)
    # The visualization run mode only shows the algorithm performing live on
    # a video. The only thing it really tests is whether your algorithm can
    # run in real-time. Its primary purpose is to provide a visualization however.
    solution.run(RunModes.VISUALIZATION, videoFilePath=os.path.abspath("./data/demovideo.avi"))
