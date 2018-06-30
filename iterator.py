import numpy as np
import cv2 as cv

def find_letterboxing(a_field,start):
    tmp_min = len(a_field[0])
    start_old = start
    for i in range(len(a_field[start])):
        if a_field[start][i] == 0:
            continue
        else:
            if tmp_min > i:
                tmp_min = i
            start -= 1
            i = 0
    i_oben_links = [start+1,tmp_min]
    start = start_old
    tmp_max = 0
    for i in range(len(a_field[start])-1,-1,-1):
        if a_field[start][i] == 0:
            continue
        else:
            if tmp_max < i:
                tmp_max = i
            start += 1
            i = len(a_field[start])-1

    return [i_oben_links,[start-1,tmp_max]]



def plate_iterator(a_map):
    points = []
    res = [[],[]]
    start = int(len(a_map[0])/2)
    for i in range(len(a_map)):
        if a_map[i][start] > 0 :
            res = find_letterboxing(a_map,i)
            i = res[1][1]
            points.append(res)
    return points
    
map = [[0,0,0,0,0],[0,0,0,0,0],[0,1,0,0,0],[0,1,1,1,0],[0,1,1,0,0],
[0,1,0,1,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

print(plate_iterator(map))
