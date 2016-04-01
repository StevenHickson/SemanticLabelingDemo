#!/usr/bin/python

caffe_root = '/home/ubuntu/git/caffe/'
#caffe_root = '/home/ubuntu/caffe-fcn/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
import numpy as np
import time

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# load net
net = caffe.Net('deploy.prototxt', 'final.caffemodel', caffe.TEST)

# Camera 0 is the integrated web cam on my netbook
camera_port = 0
camera = cv2.VideoCapture(camera_port)

def colormap(img):
  mapping = np.array([((0,0,0)),
                      ((121,181,90)),
                      ((57,114,173)),
                      ((224,82,100)),
                      ((225,2,148)),
                      ((122,117,57)),
                      ((164,116,59)),
                      ((191,189,115)),
                      ((161,26,93)),
                      ((160,41,161)),
                      ((1,10,80)),
                      ((223,217,18)),
                      ((144,244,179)),
                      ((255,215,57)),
                      ((249,188,56)),
                      ((122,42,117)),
                      ((119,60,7)),
                      ((132,211,182)),
                      ((104,130,120)),
                      ((162,67,39))])

  #mapping = colormap.values()
  #mapping = np.insert(mapping, 0, (0,0,0))
  print mapping
  img = np.uint8(img)
  result = mapping[img]
  result = np.reshape(result, (227,227,3))
  #result = np.array((227,227,3))
  #it = np.nditer(img)
  #it2 = np.nditer(result)
  #while not it.finished:
  #  print it[0]
  #  tmp = mapping[it[0]]
  #  it2[0] = tmp[0]
  #  it2.iternext()
  #  it2[0] = tmp[1]
  #  it2.iternext()
  #  it2[0] = tmp[2]
  #  it2.iternext()
  #  it.iternext()
  print result.shape
  return result

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

start = time.time()
img = cv2.resize(camera_capture, (227,227))
in_ = np.array(img, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
#print in_.shape
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()

out_probs = net.blobs['prob'].data[...]
out = out_probs[:,:].argmax(axis=1)
#print out.shape
#print np.unique(out)
seg = net.blobs['segmentations'].data[0]
seg = seg.reshape(227,227)

#Replace segId with prob argmax
result = seg
for j in range(0,out.shape[0]):
    result[seg == j] = out[j]
result = result[:,:,np.newaxis]
#print result.shape

#result = cv2.applyColorMap(result, cv2.COLORMAP_HSV);
result = colormap(result)
end = time.time()
print end - start
cv2.imwrite('output.png',result)

del(camera)
