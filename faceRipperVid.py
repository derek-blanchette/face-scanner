# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import timeit
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math
import sys
from shutil import copy


## Format seconds for time output
def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds) 
    

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')


## Get arguments 
## File name

inputfile = sys.argv[1:][1]
print('Input file is ' + os.path.join(base_dir + 'video/' + str(inputfile)))
outdir = "results/" + str(inputfile[:-4])



## Sample factor [FPS/k = how many frames are skipped between detection runs]
## Default of 1 means we sample 1 frame every second.
##            2 means we sample 2 frames every second.
##            If k = FPS, then all frames are checked for faces.
k = 1


## Check for and make output directories

# Parent sub

print("Parent sub " + os.path.join(base_dir + str(outdir)))

if not os.path.exists(os.path.join(base_dir + str(outdir))):
    os.mkdir(os.path.join(base_dir + str(outdir)))     
    
# Sub directories

print("Sub directory " + os.path.join(base_dir + str(outdir) + "/" + 'vid_faces'))

if not os.path.exists(os.path.join(base_dir + str(outdir) + "/" + 'vid_faces')):
    os.mkdir(os.path.join(base_dir +  str(outdir) + "/" + 'vid_faces'))     
   
if not os.path.exists(os.path.join(base_dir + str(outdir) + "/" + 'vid_frames')):
    os.mkdir(os.path.join(base_dir +  str(outdir) + "/" + 'vid_frames'))


# Copy file to results folder
print("Copying source video to results folder ...", end=" ")
copy(os.path.join(base_dir + 'video/' + str(inputfile)), os.path.join(base_dir + outdir))
print("Done")


# Read the model
print("Loading OpenCV Face Detection Neural Network ...", end =" ")
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
print("Done")

# initialize the video stream to get the video frames
print("Starting video stream for processing ...", end =" ")
vs = FileVideoStream(os.path.join(base_dir + 'video/' + str(inputfile))).start()
print("Done")

## Temporarily use VideoCapture to get video info
temp = cv2.VideoCapture(os.path.join(base_dir + 'video/' + str(inputfile)))

## Get FPS - frames per second
file_fps = round(temp.get(cv2.CAP_PROP_FPS))
## Get width of video file
width  = temp.get(cv2.CAP_PROP_FRAME_WIDTH)

print("Media Details: FPS=" + str(round(file_fps)) +", Width=" + str(round(width))+"px")
print("Reading " + str(k) + " decoded frame(s) for every second of video.")

## Close VideoCapture object
temp.release()

## Specify pixels to resize to before running classifier.
## If resize_to = width, then resizing is skipped
resize_to = width

## Initialize face counter
totalfaces = 0

## Done initializing video
time.sleep(1.0)

# start frame count
framenumber = 0

# start the timer
start = timeit.default_timer()


#loop the frames from the  VideoStream
while vs.more():
    #Get the frams from the video stream and resize to 400 px
    frame_orig = vs.read()
    framenumber = framenumber + 1
    
    ### Use this line to watch EVERY frame instead of just the checked frames:
    ### cv2.imshow("Frame", frame_orig)
    
    if framenumber%(file_fps/k) == 0 :
    
        print("Processing frame " +str(framenumber))
        
        try:
            if width != resize_to:
                frame = imutils.resize(frame_orig,width=resize_to)
            else:
                frame = frame_orig.copy()
        except AttributeError:
            print("~~ End of Video ~~")
        
        ### Removed the color analysis window for the sake of speed
        ### Create color information histogram
        ## tuple to select colors of each channel line
        #colors = ("r", "g", "b")
        #channel_ids = (0, 1, 2)

        ## create the histogram plot, with three lines, one for
        ## each color
        #plt.xlim([0, 256])       
        #plt.clf()
        #for channel_id, c in zip(channel_ids, colors):
        #    histogram, bin_edges = np.histogram(
        #        frame[:, :, channel_id], bins=256, range=(0, 256)
        #    )
        #    plt.plot(bin_edges[0:-1], histogram, color=c)

        #plt.xlabel("Color value")
        #plt.ylabel("Pixels")
        
        #plt.ion()
        #plt.show()
        #plt.draw()
        #plt.pause(0.001)
        
        

        ## Extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
        (h, w) = frame.shape[:2]
        # blobImage convert RGB (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # passing blob through the network to detect and prediction
        net.setInput(blob)
        detections = net.forward()
        

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence and prediction

            confidence = detections[0, 0, i, 2]

            # filter detections by confidence greater than the minimum confidence
            if confidence < 0.5 :
                continue

            # Determine the (x, y)-coordinates of the bounding box for the
            # object
            
            totalfaces = totalfaces + 1
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            the_scale = width / resize_to # factor to convert to original scale
            
            # scaled to original frame
            startX2 = math.floor(the_scale*startX)
            startY2 = math.floor(the_scale*startY)
            endX2 = math.floor(the_scale*endX)
            endY2 = math.floor(the_scale*endY)
            y2 = math.floor(the_scale*y)
            
            conf = "{:.0f}%".format(confidence * 100)
            
            # If confidence > 0.5, save face to a separate file
            if (confidence > 0.5):
                frame2 = frame_orig[startY2:endY2, startX2:endX2].copy()
                try:
                    cv2.imwrite(base_dir + str(outdir) + "/" + 'vid_faces/' + str(framenumber) + " " + str(conf) + ".jpg", frame2)
                except:
                    continue

            print("Hit: face " + str(i+1) + " with "+ "{:.1f}%".format(confidence * 100) + " on frame #" + str(framenumber) + " @ " + str(convert(round(framenumber/file_fps))))
            
            # draw the bounding box of the face along with the associated confidence
            
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        

            # draw the box on the ORIGINAL image (not resized) w/ associated confidence
            cv2.rectangle(frame_orig, (startX2, startY2), (endX2, endY2), (255, 255, 255), 2)
            cv2.putText(frame_orig, text, (startX2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                        
            # output to disk original frame with detection box        
            if confidence > 0.5:
                cv2.imwrite(base_dir + str(outdir) + "/" + 'vid_frames/' + str(framenumber)+ " " + str(conf) +".jpg", frame_orig)
            
        # show the output frame
        # cv2.imshow("Video", frame)
            
    key = cv2.waitKey(1) & 0xFF

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display elapsed time
stop = timeit.default_timer()
thetime = stop - start

print("Elapsed time: " + convert(round(thetime)))
print("Approximately " + str(totalfaces) + " total faces")

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()