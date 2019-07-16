
Project 2 - FaceSwap
--------------------

With TPS approach
-----------------
'WrapperTPS.py' contains the code to run face-swapping using the thin plate spline technique for warping


Running the program
-------------------
The program takes the following command line arguments-

--predictorpath = Path to the dlib landmarks predictor
--videofilepath = Path to video file to be edited
--image2filepath = Path to image in case of single face swap
--swapmode = Single (single face with image) or double (two faces in the same video)
--outputfilepath = Filepath to store output video
--simulateframe = To toggle frame simulation for flickering 


e.g. Swapping two faces in single video
$ python WrapperTPS.py --swapmode=2 --videofilepath='../Data/video1.mp4'

e.g. Swapping single face in video with face from image file
$ python WrapperTPS.py --swapmode=1 --videofilepath='../Data/video1.mp4' --image2filepath='../Data/TestSet_P2/Rambo.jpg'



With PRNet approach
-------------------
'WrapperPRNet.py' contains the code to run face-swapping using PRNet. The net model first has to be downloaded and placed into Code/Data/net-data/, as is mentioned on the PRNet Github.


Running the program
-------------------
The program takes the following command line arguments,

--videofilepath = Path to video that is to be modified
--ref_path = Path to image, for single face swapping face from image to a video
--outputfilepath = Path to write output video
--mode = 1: for single face swap, 2: for double face swap using faces in a single video
--gpu = to set GPU id


e.g.
$ python WrapperPRNet.py


'WrapperTri_PRN.py' also has functionality to perform the PRNet approach, but the results presented here are by using 'WrapperPRNet.py'. A separate README file has been provided for that wrapper.


