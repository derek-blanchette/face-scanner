# face-ripper - extract faces from video (both whole frames and cropped faces)

## Running the app locally
Place source video in /video subfolder.

From the project folder open CMD/Terminal and run:

`python faceRipperVid.py -i "file.mp4"`

Output is to /results in a subfolder named after the source file.
Additionally: 
+ vid_faces contains cropped faces
+ vid_frames contains whole frames from source video with detection boxes