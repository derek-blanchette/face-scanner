# face-ripper: extract faces from video

## About
Face-ripper uses the face detection deep neural network (dnn) in OpenCV to detect, label and extract faces from supported video formats. To be clear, this is an amateur Python project so expect the unexpected. 

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
