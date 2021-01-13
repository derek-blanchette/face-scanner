# face-ripper: extract faces from video

## Running the app locally
Place source video in /video subfolder.

From the project folder open CMD/Terminal and run:

`python faceRipperVid.py -i "file.mp4"`

Note:
+ Output is to a subfolder in /results named after the source file
+ vid_faces contains cropped faces
+ vid_frames contains whole frames from source video with detection boxes
