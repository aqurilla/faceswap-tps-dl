#!/usr/bin/evn python

"""
Utility functions for TPS faceswap

Author:
Nitin Suresh
ECE - UMD
"""

# Imports
import numpy as np
import cv2
import dlib
import argparse
import pdb

def shape_to_np(shape, dtype="int"):
	# Return landmarks as np-array
    # Using the 68-point landmark predictor
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def Ufunc(r):
    # U function from TPS equation
    return r*r*np.where(r==0,0.001,np.log(r))

def TPSfunc(x,y,P,soln):
    # Returns the f value calculated in TPS
    point_sum = np.zeros(x.shape)
    for (i,pi) in enumerate(P):
        point_sum += soln[i]*Ufunc(np.sqrt( (x-pi[0])**2 + (y-pi[1])**2 ))
    return (soln[-1] + soln[-2]*y + soln[-3]*x + point_sum)

def swapTPS(orig_img, final_img, from_pts, to_pts):
    """
    Swaps one face at a time
    orig_img - image containing both faces without modifications
    final_img - image to perform modifications to
    """
    # Ignore log div by 0 error
    np.seterr(divide='ignore')
    n_features = to_pts.shape[0]
    lmbda = 10

    # The original image is masked to only show the face region contained inside the convex hull
    img2 = maskConvex(orig_img, to_pts)

    # Soln contains the TPS equation coefficients for x and y coordinates
    soln = solveTPS(from_pts, to_pts, n_features, lmbda)

    # Determine region from which face should be extracted
    (nrows, ncols) = orig_img.shape[:2]

    xmin = np.min(from_pts[:,0])
    xmax = np.max(from_pts[:,0])
    ymin = np.min(from_pts[:,1])
    ymax = np.max(from_pts[:,1])

    xlin = np.arange(xmin, xmax, 1)
    ylin = np.arange(ymin, ymax, 1)

    warped_img = np.full((orig_img.shape[0], orig_img.shape[1], 3), 0, dtype=np.uint8)
    warped_mask = np.full((orig_img.shape[0], orig_img.shape[1], 3), 0, dtype=np.uint8)

    for x in xlin:
        for y in ylin:
            warp_x = TPSfunc(x, y, from_pts, soln[:,0])
            warp_y = TPSfunc(x, y, from_pts, soln[:,1])

            if warp_x>0 and warp_x<ncols and warp_y>0 and warp_y<nrows:
                if img2[int(warp_y), int(warp_x), 0] != 0:
                    warped_img[int(y), int(x), :] = img2[int(warp_y), int(warp_x), :]
                    warped_mask[int(y), int(x), :] = (255,255,255)

    # Blend the images using seamlessClone cv2 function
    center_p = (int((xmin+xmax)/2), int((ymin+ymax)/2))
    blended_img = cv2.seamlessClone(warped_img, final_img, warped_mask, center_p, cv2.NORMAL_CLONE)

    return blended_img

def maskConvex(orig_img, to_pts):
    # Find convex hull of these points to extract face region
    mask_img = np.full((orig_img.shape[0], orig_img.shape[1]), 0, dtype=np.uint8)
    hullPoints = cv2.convexHull(to_pts)
    cv2.fillConvexPoly(mask_img, hullPoints, (255,255,255))
    mask_img = np.dstack((mask_img, mask_img, mask_img))

    return cv2.bitwise_and(orig_img, mask_img)

def solveTPS(from_pts, to_pts, n_features, lmbda):
    # Formulate TPS equation, follows notation from project statement
    K = np.zeros((n_features, n_features), dtype=np.float32)
    P = np.ones((n_features, 3), dtype=np.float32)
    V = np.zeros((n_features+3, 2), dtype=np.float32)

    xarr = np.subtract.outer(from_pts[:,0], from_pts[:,0])
    yarr = np.subtract.outer(from_pts[:,1], from_pts[:,1])
    K = Ufunc(np.sqrt(xarr**2 + yarr**2))

    # Pi = [xi, yi, 1]
    P[:,:2] = from_pts

    # V is the required output, i.e. to_pts
    V[:n_features] = to_pts

    K_mat_p1 = np.concatenate((K, P), axis=1)
    K_mat_p2 = np.concatenate((np.transpose(P), np.zeros((3,3), dtype=np.float32)), axis=1)

    K_mat = np.concatenate((K_mat_p1, K_mat_p2), axis=0)

    iden_mat = lmbda*np.identity(n_features+3)

    inv_mat = np.linalg.pinv(K_mat+iden_mat)

    # Solve TPS for x-coordinates and y-coordinates
    return np.dot(inv_mat, V)

def dispLandmarks(img, rects, predictor):
    # Display facial landmarks in the image
    landmarks_img = img
    for (i, rect) in enumerate(rects):
        shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rect)
        shape = shape_to_np(shape)

        (x,y,w,h) = rect_to_bb(rect)
        if i==0:
            cv2.rectangle(landmarks_img, (x,y), (x+w, y+h), (0,255,0), 2)
        else:
            cv2.rectangle(landmarks_img, (x,y), (x+w, y+h), (255,0,0), 2)

        for (x,y) in shape:
            cv2.circle(landmarks_img, (x,y), 1, (0,0,255), -1)
    cv2.imshow("Detected landmarks", landmarks_img)

def imgResize(img, fcols=512):
    # Resize image to have specified width fcols
    (rows, cols) = img.shape[:2]
    frows = int((fcols/cols)*rows)
    return cv2.resize(img, (fcols, frows), interpolation=cv2.INTER_CUBIC)

def processPoints(to_pts, from_pts, to_ptsArray, from_ptsArray, n_features, dwnsamp_factor, idx):
    # Convert to np-arrays and downsample
    to_pts = shape_to_np(to_pts)
    from_pts = shape_to_np(from_pts)

    to_ptsArray = np.append(to_ptsArray, to_pts.reshape((1,n_features*2)), axis=0)
    from_ptsArray = np.append(from_ptsArray, from_pts.reshape((1,n_features*2)), axis=0)

    if idx>2:
        to_ptsArray = to_ptsArray[-2:,:]
        inter_to_pts = np.mean(to_ptsArray, axis=0, dtype=int).reshape((n_features, 2))
        from_ptsArray = from_ptsArray[-2:,:]
        inter_from_pts = np.mean(from_ptsArray, axis=0, dtype=int).reshape((n_features, 2))
        inter_to_pts = inter_to_pts[::dwnsamp_factor]
        inter_from_pts = inter_from_pts[::dwnsamp_factor]
    else:
        inter_to_pts = to_pts[::dwnsamp_factor]
        inter_from_pts = from_pts[::dwnsamp_factor]

    to_pts = to_pts[::dwnsamp_factor]
    from_pts = from_pts[::dwnsamp_factor]
    return (to_pts, from_pts, to_ptsArray, from_ptsArray, inter_to_pts, inter_from_pts)
