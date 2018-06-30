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

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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
        ncolor = 50

        img = frame
        img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

        scale = 1 + (len(img[0]) / 1500)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (int(3), int(3)), 0)
        img = cv.Sobel(img, -1, 1, 0)

        img = np.multiply(np.floor(np.divide(img, ncolor)), ncolor).astype(np.uint8)
        h, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        se = cv.getStructuringElement(cv.MORPH_RECT, (int(16 * scale), int(4 * scale)))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, se)

        el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(5 * scale), int(5 * scale)))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, el)

        (cont_img, cnts, hierarchy) = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # loop over our contours
        candidates = []

        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) < 8 and len(approx) > 3:
                pnts = cv.convexHull(approx)

                x = []
                y = []

                x2 = []
                y2 = []

                for p in pnts:
                    x.append(p[0][0])
                    y.append(p[0][1])

                for p in approx:
                    x2.append(p[0][0])
                    y2.append(p[0][1])

                A = PolyArea(x, y)
                A2 = PolyArea(x2, y2)

                # FILTER BY AREA
                if A / A2 < 1.2:
                    rect = cv.minAreaRect(approx)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    edges = []
                    for i in range(len(box) - 1):
                        edges.append(np.sqrt(np.power(box[i][0] - box[i + 1][0], 2) + np.power(
                            box[i][0] - box[i + 1][0], 2)))
                    edges.append(np.sqrt(np.power(box[3][0] - box[0][0], 2) + np.power(
                        box[3][0] - box[0][0], 2)))

                    sorted_arr = sorted(edges, reverse=True)

                    # check: if any edge == 0
                    if True or len(np.array(edges).nonzero()[0]) == len(edges):
                        # check: two largest edges approx same length
                        if sorted_arr[0] / sorted_arr[1] < 1.05:
                            # check: connected edges have larger different length
                            if max(edges[0], edges[1]) / min(edges[0], edges[1]) > 3 and max(edges[2], edges[1]) / min(
                                    edges[2], edges[1]) > 3:
                                # check: third larges edge much smaller
                                candidates.append(pnts)
                                if sorted_arr[1] / sorted_arr[2] > 2 and sorted_arr[1] / sorted_arr[3] > 2:
                                    print(3)

                                    cv.drawContours(img, [box], -1, 50, 3)

                                    xlen = len(img[0])
                                    ylen = len(img)
                                    box = [[d[0] / xlen, d[1] / ylen] for d in box]
                                    return box


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
