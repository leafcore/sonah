from api.hackathon import HackathonApi, RunModes
import os
import cv2
import matplotlib.pyplot as plt
from tesserwrap import Tesseract
from PIL import Image
import numpy as np
import itertools


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class Solution(HackathonApi):
    def handleFrameForTaskB(self, frame, regionCoordinates):
        try:
            coordinates = list()
            for point in regionCoordinates[0]["coordinates"]:
                coordinates.append([point[0]*frame.shape[1], point[1]*frame.shape[0]])
            coordinates = np.int0(coordinates)

            frame = cv2.drawContours(frame, [coordinates], 0, (0, 255, 0), 2)
            #cv2.imshow("test", frame)
            #cv2.waitKey(0)
            #plt.imshow(frame)
            #plt.show()
            warped = four_point_transform(frame, coordinates)

            shrunk = cv2.cvtColor(warped[:, int(warped.shape[1]/10):], cv2.COLOR_BGR2GRAY)
            #plt.imshow(shrunk)
            #plt.show()

            scale = 6
            shrunk = cv2.resize(shrunk, (shrunk.shape[1]*scale,shrunk.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
            #plt.imshow(shrunk)
            #plt.show()
            _, shrunk = cv2.threshold(shrunk, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #plt.imshow(shrunk)
            #plt.show()
            shrunk = 255-cv2.dilate(255-shrunk, np.ones((1, 1)), iterations=1)
            #plt.imshow(shrunk)
            #plt.show()

            #cv2.imshow("shrunk", shrunk)
            #qqqcv2.waitKey(0)
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
          
                plt.imshow(snip)
                plt.show()

                im = Image.fromarray(np.uint8(snip/np.max(snip)*255))
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
                print(plate)
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
            print(exception)
            return None


if __name__ == "__main__":
    solution = Solution()
    datasetWrapper = solution.initializeApi(os.path.abspath("./metadata.json"), os.path.abspath("./data/"))
    for i in range(20):
        frame = datasetWrapper.getFrame(i)
        rois = datasetWrapper.getRois(i)
        res = solution.handleFrameForTaskB(frame, rois)
        print(res)
    id = 16#2
    print(solution.handleFrameForTaskB(datasetWrapper.getFrame(id), datasetWrapper.getRois(id)))
    plt.imshow(datasetWrapper.getFrame(id))
    plt.show()


