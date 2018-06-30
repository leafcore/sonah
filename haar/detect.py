import cv2 as cv
import numpy as np
import glob

folderpath = "/home/bonzai/Repos/sonah_hackathon/data/work/multiple street"


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def plate_detection(folderpath, ncolor=50):
    for fn in glob.glob(folderpath + "/*")[::]:

        # sharpening
        img = cv.imread(fn)
        img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
        # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
        # img = cv.filter2D(img, -1, kernel)

        scale = 1 + (len(img[0]) / 1500)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (int(3), int(3)), 0)

        # blur = cv.GaussianBlur(img, (0,0), 1)
        # img = cv.addWeighted(blur, 1.5, img, -0.5, 0)

        img = cv.Sobel(img, -1, 1, 0)

        img = np.multiply(np.floor(np.divide(img, ncolor)), ncolor).astype(np.uint8)
        h, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, , 5)
        # img = cv.threshold(img, min(mean+50, 200), 255, cv.THRESH_BINARY)[1]
        # cv.imshow("Result", img)
        # cv.waitKey(0)

        se = cv.getStructuringElement(cv.MORPH_RECT, (int(16 * scale), int(4 * scale)))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, se)

        el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(5 * scale), int(5 * scale)))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, el)

        # cv.imshow("Result", img)
        # cv.waitKey(0)

        # img = cv.threshold(img, min(mean+50, 200), 255, cv.THRESH_BINARY)[1]

        # img = np.multiply(np.floor(np.divide(img, ncolor)), ncolor).astype(np.uint8)

        # cv.threshold()

        # img = cv.Sobel(img, -1, 1, 0)

        # separate color

        # img = smooth

        # cv.imshow("Result", img)
        # cv.waitKey(0)
        # kernel = np.ones((2, 2), np.uint8)
        # edg_dil = cv.dilate(img, kernel, iterations=3)
        # img = cv.erode(edg_dil, kernel, iterations=3)
        # cv.imshow("Result", img)
        # cv.waitKey(0)

        # edge
        # blur = cv.GaussianBlur(img, (0,0), 3)
        # img = cv.addWeighted(blur, 1.5, img, -0.5, 0)

        # img = cv.threshold(img, min(mean+50, 200), 255, cv.THRESH_BINARY)[1]
        # img = cv.threshold(img, (150), 255, cv.THRESH_BINARY)[1]
        # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)

        # blur = cv.GaussianBlur(img, (3,3), 3)
        # img = cv.addWeighted(blur, 1.5, img, -0.5, 0)
        # img = cv.Canny(img, mean -50, mean+50, apertureSize=3)

        # kernel = np.ones((2, 2), np.uint8)
        # edg_dil = cv.dilate(img, kernel, iterations=3)
        # img = cv.erode(edg_dil, kernel, iterations=3)
        # img = cv.threshold(img, (mean), 255, cv.THRESH_BINARY)[1]

        (cont_img, cnts, hierarchy) = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
        ## img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2)

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
                            if max(edges[0], edges[1]) / min(edges[0], edges[1]) > 3 and max(edges[2], edges[1]) / min(edges[2], edges[1]) > 3:
                                # check: third larges edge much smaller
                                candidates.append(pnts)
                                if sorted_arr[1] / sorted_arr[2] > 2 and sorted_arr[1] / sorted_arr[3] > 2:
                                    print(3)

                                    cv.drawContours(img, [box], -1, 50, 3)

        cv.imshow("Result", img)
        cv.waitKey(0)


plate_detection(folderpath)

