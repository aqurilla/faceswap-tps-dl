#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 2: FaceSwap Phase 1 Wrapper code

References:
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Author:
Nitin Suresh
ECE - UMD
"""

# Code starts here:

# Imports
import numpy as np
import cv2
import dlib
import argparse
import pdb
from utils import *

def main():
	# Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--predictorpath', default='../Model/shape_predictor_68_face_landmarks.dat', help='dlib face landmarks predictor')
    Parser.add_argument('--videofilepath', default='../Data/video1.mp4', help='Path to video file')
    Parser.add_argument('--image2filepath', default='../Data/TestSet_P2/Rambo.jpg', help='Path to image file for single faceswap')
    Parser.add_argument('--swapmode', default=2, help='Mode for performing swap: 1 for single faceswap, 2 for double faceswap')
    Parser.add_argument('--outputfilepath', default='../Output/doublefaceswap.avi', help='Path to write output video')
    Parser.add_argument('--simulateframe', default=0, help='1: Frame simulation to improve flickering, 0: No frame simulation')

    Args = Parser.parse_args()
    predictorpath = Args.predictorpath
    videofilepath = Args.videofilepath
    image2filepath = Args.image2filepath
    swapmode = int(Args.swapmode)
    outputfilepath = Args.outputfilepath
    simulateframe = int(Args.simulateframe)

    # Setup dlib detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorpath)

    # Specify factor by which to downsample landmarks points (originally 68 points)
    dwnsamp_factor = 2
    n_features = 68
    if simulateframe:
        fps_rate = 60
    else:
        fps_rate = 30

    # Initialize video reader and writer objects
    cap = cv2.VideoCapture(videofilepath)
    ret, img = cap.read()
    resized_img = imgResize(img, 512)

    # Rotate image if required
    # resized_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)

    (frame_height,frame_width) = resized_img.shape[:2]
    out = cv2.VideoWriter(outputfilepath, cv2.VideoWriter_fourcc('M','J','P','G'), fps_rate, (frame_width,frame_height))

    # For frame simulation
    idx=0
    to_ptsArray = np.zeros((1,n_features*2))
    from_ptsArray = np.zeros((1,n_features*2))


    while(cap.isOpened()):
        ret, img = cap.read()

        if ret==True:
            idx += 1

            # Resize image to specified width
            img = imgResize(img, 512)

            # Rotate image if required
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            if swapmode==1:
                """
                Swap a single face from the video, with a face from a second image
                """
                cimg = cv2.imread(image2filepath)

                # Resize image to specified width
                cimg = imgResize(cimg, 512)

                # Detect faces in both the images
                rects1 = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
                rects2 = detector(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), 1)

                # Swap if atleast 1 face detected in both images
                if rects1 and rects2:
                    to_pts = predictor(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), rects2[0])
                    from_pts = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rects1[0])

                    to_pts, from_pts, to_ptsArray, from_ptsArray, inter_to_pts, inter_from_pts = \
                    processPoints(to_pts, from_pts, to_ptsArray, from_ptsArray, n_features, dwnsamp_factor, idx)

                    # Swap face from second image into the first image
                    output_img = swapTPS(cimg, img, from_pts, to_pts)

                    if idx>2 and simulateframe:
                        # Simulating frames for reducing flickering
                        intermediate_img = swapTPS(cimg, img, inter_from_pts, inter_to_pts)
                else:
                    # If both faces are not detected, return original image
                    output_img = img
                    if idx>2 and simulateframe:
                        intermediate_img  = img
            else:
                """
                Swap both faces in a single image
                """
                # Detect faces in the image
                rects = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
                # dispLandmarks(img, rects, predictor)

                # Perform faceswap only if more than 1 face is detected in the image
                if len(rects)>1:
                    to_pts = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rects[0])
                    from_pts = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rects[1])

                    to_pts, from_pts, to_ptsArray, from_ptsArray, inter_to_pts, inter_from_pts = \
                    processPoints(to_pts, from_pts, to_ptsArray, from_ptsArray, n_features, dwnsamp_factor, idx)

                    # Perform one swap at a time, interchanging the points for the second swap
                    output1 = swapTPS(img, img, from_pts, to_pts)
                    output_img = swapTPS(img, output1, to_pts, from_pts)

                    if idx>2 and simulateframe:
                        # Simulating frames for reducing flickering
                        intermediate_img1 = swapTPS(img, img, inter_from_pts, inter_to_pts)
                        intermediate_img = swapTPS(img, intermediate_img1, inter_to_pts, inter_from_pts)

                else:
                    # If only 1 face detected, return original image
                    output_img = img
                    if idx>2 and simulateframe:
                        intermediate_img  = img

            """
            Filtering to improve motion flickering
            """
            # Blur image to reduce flickering
            output_img = cv2.bilateralFilter(output_img, 9, 75, 75)

            if idx>2 and simulateframe:
                intermediate_img = cv2.bilateralFilter(intermediate_img, 9, 75, 75)

            """
            Display output image and write to file
            """
            # Display input image and final output image
            cv2.imshow("Original image", img)
            if idx>2 and simulateframe:
                cv2.imshow("Intermediate image", intermediate_img)
            cv2.imshow("Output image", output_img)
            cv2.waitKey(25)

            # Write both frames to video
            if idx>2 and simulateframe:
                out.write(intermediate_img)
            out.write(output_img)

        else:
            # When the video finishes
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
