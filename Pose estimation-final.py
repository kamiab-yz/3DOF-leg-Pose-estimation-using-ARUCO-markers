#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:12:23 2021

@author: kamiab-yz
"""


import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import time
import os
###################################################################################### REAL SENSE MATRIX
#matrix_coefficients= np.matrix(  [[604.98413094,   0.,         319.58236873],
# [  0.,         606.33367394, 236.49601336],
 #[  0.,           0.,           1.        ]])

#distortion_coefficients = np.matrix([[ 4.72376758e-02,  8.02417817e-01,  1.08129524e-03, -1.02383570e-03,
 # -3.56428345e+00]])
######################################################################################################

#################################################################################3MOBILE
matrix_coefficients=np.matrix([[3.07951233e+03, 0.00000000e+00, 1.54855596e+03] ,
 [0.00000000e+00, 3.10799290e+03, 2.02073037e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.matrix([[ 0.13957676, -0.89879243,  0.00197684, -0.00455686,  0.86659383]])

##    objp   topl,topr,downr,downl   order: 0 3 6 9   or    2 5 8 11
objp = np.array([#topl
 [0., 0.12, 0.],
 [0., 0.088, 0.],
 [0., 0.056, 0.],
 [0., 0.024, 0.],
 #topr
 [0.024, 0.12, 0.],
 [0.024, 0.088, 0.],
 [0.024, 0.056, 0.],
 [0.024, 0.024, 0.],
 #downr
 [0.024, 0.096, 0.],
 [0.024, 0.068, 0.],
 [0.024, 0.032, 0.],
 [0.024, 0., 0.],
 #downl
 [0., 0.096, 0.],
 [0., 0.068, 0.],
 [0., 0.032, 0.],
 [0., 0., 0.]])
#print("objp:", objp)

objp_m = np.array([  # topl  order  15 18 0 3 6 9
    [-0.00406, 0.056, -0.02758],
    [-0.00406, 0.024, -0.02758],
    #[0., 0.12, 0.],
    #[0., 0.088, 0.],
    [0., 0.056, 0.],
    [0., 0.024, 0.],
    # topr
    [-0.00406, 0.056, -0.00358],
    [-0.00406, 0.024, -0.00358],
    #[0.024, 0.12, 0.],
    #[0.024, 0.088, 0.],
    [0.024, 0.056, 0.],
    [0.024, 0.024, 0.],
    # downr
    [-0.00406, 0.032, -0.00358],
    [-0.00406, 0., -0.00358],
    #[0.024, 0.096, 0.],
    #[0.024, 0.068, 0.],
    [0.024, 0.032, 0.],
    [0.024, 0., 0.],
    # downl
    [-0.00406, 0.032, -0.02758],
    [-0.00406, 0., -0.02758],
    #[0., 0.096, 0.],
    #[0., 0.068, 0.],
    [0., 0.032, 0.],
    [0., 0., 0.]])



def inversePerspective(rvec, tvec):
    #               inverse the rvec and tvec
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec






def findArucoMarkers(img , markerSize=4 , totalMarkers=50, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global matrix_coefficients
    global distortion_coefficients
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParam = aruco.DetectorParameters_create()
    bbox, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam, cameraMatrix = matrix_coefficients,
                                                                distCoeff = distortion_coefficients)

    return bbox, ids

def FindIndex(want,ids):
    #                                  pass the index of the want id
    for i in range(len(ids)):
        if ids[i] == want:
            return i

def SortMarkers(ids, bbx, coordinate):
    #               make list of corners in order of the markers example: top_l: topleft of marker numbers 0 3 6 9 15 18

    #list = [0, 3, 6, 9, 15, 18]
    if coordinate == 'm':
        list = [15, 18, 6, 9]
    elif coordinate == 'w':
        list = [2, 5, 8, 11]
    top_l = []
    top_r = []
    down_r = []
    down_l = []
    for i, value in enumerate(list):
        index = FindIndex(value, ids)
        top_l.append(bbx[index][0][0])
        top_r.append(bbx[index][0][1])
        down_r.append(bbx[index][0][2])
        down_l.append(bbx[index][0][3])
    return top_l, top_r, down_r, down_l


def findCenter(w_top_l, w_top_r, w_down_r, w_down_l):
    center2 = (w_top_l[0] + w_top_r[0] + w_down_l[0] + w_down_r[0])/4
    center5 = (w_top_l[1] + w_top_r[1] + w_down_l[1] + w_down_r[1])/4
    return np.array([center2]), np.array([center5])

def draw(img, corners, imgpts):

    corner = tuple(corners[-1].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
    return img


# End points of coordinate axes:
axis = np.float32([[0.024, 0, 0], [0, 0.024, 0], [0, 0, 0.024]]).reshape(-1, 3)
axis2 = np.float32([[0.024, 0.032, 0], [0, 0.056, 0], [0.0, 0.032, 0.024]]).reshape(-1, 3)

point_test = np.float32([0.036, 0.012, 0.0])
tip_point = np.float32([0.045, -0.071, -0.012])
#tip_point = np.float32([0.0, -0.0, -0.0])
# termination criteria
# In this case the maximum number of iterations is set to 30 and epsilon = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
tip_x = []
tip_y = []
tip_z = []

t_x = []
t_y = []
t_z = []
time_list=  []

robot_posX = []
robot_posY = []
robot_posZ = []

#error list
e_x = []
e_y = []
e_z = []

refZ = []
refY = []

def main():
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('mobile-new.mp4')
    cap = cv2.VideoCapture('1.mp4')
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('final-video-out.mp4', fourcc, 30.0, (1920, 1080))
    while True:

        ret, img = cap.read()
        if ret != 0:
            bbx, ids = findArucoMarkers(img)
            aruco.drawDetectedMarkers(img, bbx, ids)  # Draw A square around the markers and their ids
            if len(ids) == 10:
               # print ("ids:", ids)
                top_l, top_r, down_r, down_l = SortMarkers(ids, bbx, 'm')
                w_top_l, w_top_r, w_down_r, w_down_l = SortMarkers(ids, bbx, 'w')


                #cv2.circle(img, tuple(center2), 8, (0, 128, 255), -1)

                #   make the arrays of 3D points in model and world coordinates
                points2D = np.concatenate((top_l, top_r, down_r, down_l), axis=0) #   corners point of the  0 3 6 9 12 markers in model coordinate
                world_points2D = np.concatenate((w_top_l, w_top_r, w_down_r, w_down_l), axis=0)

                # SOLVEPNP EQ
                _, rvec, tvec= cv2.solvePnP(objp_m, points2D, matrix_coefficients, distortion_coefficients)#, flags=cv2.CV_ITERATIVE)
                _, w_rvec, w_tvec = cv2.solvePnP(objp, world_points2D, matrix_coefficients, distortion_coefficients)  # , flags=cv2.CV_ITERATIVE)

                #flag, w_rvec, w_tvec, inliers = cv2.solvePnPRansac(w_objp, world_points2D, matrix_coefficients, distortion_coefficients)

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, matrix_coefficients, distortion_coefficients)
                w_imgpts, jac1 = cv2.projectPoints(axis, w_rvec, w_tvec, matrix_coefficients, distortion_coefficients)
                end_point, jac = cv2.projectPoints(tip_point, rvec, tvec, matrix_coefficients, distortion_coefficients)
                point_11, jac = cv2.projectPoints(point_test, w_rvec, w_tvec, matrix_coefficients, distortion_coefficients)

                # Draw coordinate axes (3 lines) on the image
                img = draw(img, points2D, imgpts)
                img = draw(img, world_points2D, w_imgpts)

                # Draw tip point
                cv2.circle(img, tuple(end_point[0][0]), 8, (0, 0, 255), -1)
                cv2.circle(img, tuple(point_11[0][0]), 8, (255, 0, 255), -1)

                #   calculate the relative position between world and model coordinates
                firstRvec = rvec
                firstTvec = tvec

                secondRvec = w_rvec
                secondTvec = w_tvec



                firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
                secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))

                composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)

                print ("relative trans:", composedTvec)

                # transformation matrix
                R, _ = cv2.Rodrigues(composedRvec)
                print("R=", R)
                print("t=",composedTvec)
                H_tr = np.concatenate((R, composedTvec), axis=1)
                hemog = np.array([0, 0, 0, 1]).reshape(1, 4)
                H_tr = np.concatenate((H_tr, hemog), axis=0)
                print ("H:",H_tr)

                #transform point to world axis
                tip_point_hemog = np.append(tip_point, np.array([1]), axis=0)

                print("t:", tip_point_hemog)
                tip_point_world = np.dot(H_tr, tip_point_hemog)

                print("tip", tip_point_world)

                # error calculation compare to rbdl outputs
                #x_robot = tip_point_world[2] + 0.1
                #y_robot = (-tip_point_world[0]) - 0.007
                #z_robot = (tip_point_world[1] - 0.056)

                x_robot = (- tip_point_world[0]) - 0.007
                y_robot = (tip_point_world[2]) + 0.1
                z_robot = (tip_point_world[1] - 0.056)




                # project the tip point with xyz of the world we found and w_rec w_tvec
                test1 = np.float32([tip_point_world[0], tip_point_world[1], tip_point_world[2]]) # get rid of hemog
                test2, jac = cv2.projectPoints(test1, w_rvec, w_tvec, matrix_coefficients, distortion_coefficients)
                cv2.circle(img, tuple(test2[0][0]), 8, (0, 255, 255), -1)
                #print ("relative rot:",composedRvec)

                # data saving
                tip_x.append(tip_point_world[0])
                tip_y.append(tip_point_world[1])
                tip_z.append(tip_point_world[2])

                t_x.append(composedTvec[0])
                t_y.append(composedTvec[1])
                t_z.append(composedTvec[2])
                time_list.append(time.time())


                robot_posX.append(x_robot)
                robot_posY.append(y_robot)
                robot_posZ.append(z_robot)

                #e_x.append(x_robot - 0.078 )
                e_y.append(y_robot - 0.078)
                e_z.append(z_robot - (-0.3445))

                # image display
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 3000, 2000)
                out.write(img)
                cv2.imshow("image", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    for i in range(len(robot_posZ)):
                        refY.append(0.078)
                        refZ.append(-0.39)

                    #     PLOT DATA
                    plt.figure()
                    plt.plot(np.arange(len(tip_x)), tip_x, 'r-', label= 'X')
                    plt.plot(np.arange(len(tip_y)), tip_y, 'g-', label= 'y')
                    plt.plot(np.arange(len(tip_z)), tip_z, 'b-', label= 'z')
                    plt.legend(loc='upper right')
                    plt.title(' position of tip in world coordinate')

                    plt.figure()
                    plt.plot(np.arange(len(robot_posX)), robot_posX, 'r-', label= 'X')
                    plt.plot(np.arange(len(robot_posY)), robot_posY, 'b-', label = 'Y')
                    plt.plot(np.arange(len(robot_posZ)), robot_posZ, 'g-', label = 'Z')
                    plt.plot(np.arange(len(robot_posY)), refY , '--b', label='ref Y')
                    plt.plot(np.arange(len(robot_posZ)), refZ, '--g', label='ref Z')

                    plt.legend(loc = 'upper right')
                    plt.title('POSITION OF ROBOT IN RBDL COORDINATE ')

                    #plt.figure()
                    #plt.plot(np.arange(len(e_x)), e_x, 'r-')
                    #plt.plot(np.arange(len(e_y)), e_y, 'g-', label ='y_error')
                    #plt.plot(np.arange(len(e_z)), e_z, 'b-', label = 'z_error')
                    #plt.legend(loc= 'upper right')
                    #plt.title('ERROR ')

                    plt.show()
                    break
        else:
            cv2.destroyAllWindows()
            for i in range(len(robot_posZ)):
                refY.append(0.078)
                refZ.append(-0.39)
            plt.figure()
            plt.plot(np.arange(len(tip_x)), tip_x, 'r-', label='X')
            plt.plot(np.arange(len(tip_y)), tip_y, 'g-', label='y')
            plt.plot(np.arange(len(tip_z)), tip_z, 'b-', label='z')
            plt.legend(loc='upper right')
            plt.title(' position of tip in world coordinate')

            plt.figure()
            plt.plot(np.arange(len(robot_posX)), robot_posX, 'r-', label='X')
            plt.plot(np.arange(len(robot_posY)), robot_posY, 'b-', label='Y')
            plt.plot(np.arange(len(robot_posZ)), robot_posZ, 'g-', label='Z')
            plt.plot(np.arange(len(robot_posY)), refY, '--b', label='ref Y')
            plt.plot(np.arange(len(robot_posZ)), refZ, '--g', label='ref Z')
            plt.legend(loc='upper right')

            plt.title('POSITION OF ROBOT IN RBDL COORDINATE ')

            #plt.figure()
            # plt.plot(np.arange(len(e_x)), e_x, 'r-')
            #plt.plot(np.arange(len(e_y)), e_y, 'g-', label='y_error')
            #plt.plot(np.arange(len(e_z)), e_z, 'b-', label='z_error')
            #plt.legend(loc='upper right')
            #plt.title('ERROR ')

            plt.show()
            break


if __name__ == "__main__" :
    main()









