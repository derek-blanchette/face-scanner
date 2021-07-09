# face-ripper: extract faces from video

## About
Face-ripper uses the face detection deep neural network (dnn) in OpenCV to detect, label and extract faces from supported video formats. To be clear, this is an hobby Python project. 

## Running the app locally
Place source video in `video` subfolder.

From the project folder open CMD/Terminal and run:

`python face-ripper.py -i "file.mp4"`

Note:
+ Output is to a subfolder in `results` named after the source file
+ `vid_faces` contains cropped faces
+ `vid_frames` contains whole frames from source video with detection boxes

Example uses:
+ Catching terrorists who post video of their crimes to social media
+ Scanning all of the archived videos from Parler (~60TB)

## Limitations
The documentation is sparse and takes the form of notes to myself.
Arguments are not implemented correctly which prevents me from creating more options.
You have to edit the code if you want to change the speed at which it samples from the video (once per second, three times per second, once every 5 seconds, etc). This is controlled by the integer k. 
You have to edit the code if you want to enable the video window in order to watch the analysis occur.
