'''
Adapted from https://github.com/YadiraF/PRNet
'''

import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from misc.api import PRN
from misc.render import render_texture
import cv2
import pdb

def texture_editing(prn, image, ref_image, mode, d, d_ref):

    [h, w, _] = image.shape

    """
    Original texture
    """
    #-- 1. 3d reconstruction -> get texture.

    pos = prn.process(image, d)
    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    """
    Reference texture
    """
    #-- 2. Texture Editing
    # For double faceswap edit here to return ref_pos from the same image

    ref_pos = prn.process(ref_image, d_ref)
    ref_image = ref_image/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture#(texture + ref_texture)/2.

    """
    Swap and blend image
    """
    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    # For double swap use this output as the image for second run,
    # and use copy of reference image
    return output

    # save output
    # imsave(output_path, output)
    # print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    #parser.add_argument('-i', '--image_path', default='TestImages/testvid2.jpg', type=str,
    #                    help='path to input image') # image containing face to swap out
    parser.add_argument('-r', '--ref_path', default='../Input/Rambo.jpg', type=str,
                        help='path to reference image(texture ref)') # image containing face to swap in
    #parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str,
    #                    help='path to save output')
    parser.add_argument('--mode', default=1, type=int,
                        help='1: single face swap, 2: double face swap')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--videofilepath', default='../Input/Test1.mp4', help='Path to video file')
    parser.add_argument('--outputfilepath', default='../Output/test1_output.avi', help='Path to write output video')

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)

    # CHANGED - taking input from video
    # image = imread(parser.parse_args().image_path)
    # texture from another image or a processed texture
    ref_image = imread(parser.parse_args().ref_path)
    mode = int(parser.parse_args().mode)
    #output_path = parser.parse_args().output_path
    videofilepath = parser.parse_args().videofilepath
    outputfilepath = parser.parse_args().outputfilepath

    # Try with conversion
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    # Initialize video reader and writer objects
    cap = cv2.VideoCapture(videofilepath)
    ret, image = cap.read()

    (frame_height,frame_width) = image.shape[:2]
    out = cv2.VideoWriter(outputfilepath, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    idx = 0

    if mode==1:
        # Get reference pos for single faceswap
        ref_detected_faces = prn.dlib_detect(ref_image)
        d_ref = ref_detected_faces[0].rect

    while(cap.isOpened()):
        ret, image = cap.read()

        if ret==True:
            if mode==1:
                detected_faces = prn.dlib_detect(image)

                if len(detected_faces)==0:
                    # If no face is detected return the original image
                    output = image
                else:
                    d1 = detected_faces[0].rect
                    # d2 = detected_faces[1].rect

                    # output1 = texture_editing(prn, image, image, mode, output_path, d1, d2)
                    output = texture_editing(prn, image, ref_image, mode, d1, d_ref)
            else:
                detected_faces = prn.dlib_detect(image)

                if len(detected_faces) < 2:
                    # If only 1 face is detected return the original image
                    output = image
                else:
                    d1 = detected_faces[0].rect
                    d2 = detected_faces[1].rect

                    output1 = texture_editing(prn, image, image, mode, d1, d2)
                    output = texture_editing(prn, output1, image, mode, d2, d1)


            out.write(output)
            print('Frame number: '+str(idx))
            idx+=1
            # cv2.imshow("Inputimage", image)
            # cv2.imshow("Output image", output)
            # cv2.waitKey(25)
            # imsave(output_path, output)
            # print('Done.')

        else:
            break

    print('Done')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
