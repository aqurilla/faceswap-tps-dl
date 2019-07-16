
# FaceSwap using Thin Plate Splines and Neural Nets

This project implements face-swapping using two approaches - a traditional geometric approach using thin plate splines for face warping, and secondly using [PRNet](https://github.com/YadiraF/PRNet).

![rambo](https://github.com/aqurilla/faceswap-tps-dl/blob/master/Data/images/ramboTPS1.jpg)

## With TPS approach

`WrapperTPS.py` contains code to run face-swapping using the thin plate spline technique for warping

### Running the program

The program takes the following command line arguments-

```
--predictorpath = Path to the dlib landmarks predictor
--videofilepath = Path to video file to be edited
--image2filepath = Path to image in case of single face swap
--swapmode = Single (single face with image) or double (two faces in the same video)
--outputfilepath = Filepath to store output video
--simulateframe = To toggle frame simulation for flickering
```

e.g. Swapping two faces in single video
`$ python WrapperTPS.py --swapmode=2 --videofilepath='../Data/video1.mp4'`

e.g. Swapping single face in video with face from image file
`$ python WrapperTPS.py --swapmode=1 --videofilepath='../Data/video1.mp4' --image2filepath='../Data/TestSet_P2/Rambo.jpg'`

## With PRNet approach

`WrapperPRNet.py` contains code to run face-swapping using PRNet. The net model first has to be downloaded and placed into `Code/Data/net-data/`, as mentioned on the [PRNet page](https://github.com/YadiraF/PRNet).

### Running the program

The program takes the following command line arguments,

```
--videofilepath = Path to video that is to be modified
--ref_path = Path to image, for single face swapping face from image to a video
--outputfilepath = Path to write output video
--mode = 1: for single face swap, 2: for double face swap using faces in a single video
--gpu = to set GPU id
```

e.g.
`$ python WrapperPRNet.py`
