import cv2
import numpy as np

# Camera 0 is the integrated web cam on my netbook
camera_port = 0
camera = cv2.VideoCapture(camera_port)

# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im
 
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(30):
 temp = get_image()
print("Taking image...")
# Take the actual image we want to keep
camera_capture = get_image()
print np.array(camera_capture).shape

