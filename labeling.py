#!/usr/bin/python
import cv2
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import time
caffe_root = '/home/ubuntu/git/caffe/'
#caffe_root = '/home/ubuntu/caffe-fcn/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# load net
net = caffe.Net('deploy.prototxt', 'final.caffemodel', caffe.TEST)


capture=None

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

dataset_labels = np.array(['background',
                    'books',
                    'cabinets',
                    'ceiling',
                    'chair',
                    'computer',
                    'cup',
                    'door',
                    'fire_extinguisher',
                    'floor',
                    'fridge',
                    'keyboard',
                    'monitor',
                    'person',
                    'poster',
                    'signs',
                    'table',
                    'trashcan',
                    'walls',
                    'whiteboard'])

def create_legend(colors, pairs):
  h = 50
  w = 120
  im = np.zeros((h, colors.shape[0]*w, 3), dtype='uint8')
  for c_idx, c in enumerate(colors):
    im[:, c_idx*w : (c_idx+1)*w, 0] = colors[c_idx, 0]
    im[:, c_idx*w : (c_idx+1)*w, 1] = colors[c_idx, 1]
    im[:, c_idx*w : (c_idx+1)*w, 2] = colors[c_idx, 2]
    cv2.putText(im, pairs[c_idx], (c_idx*w+5,25),
        cv2.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255])
  return im

def create_legend_image(colors,pairs):
  result1 = create_legend(colors[:10],pairs[:10])
  result2 = create_legend(colors[10:],pairs[10:])
  result = np.concatenate((result1,result2),axis=0)
  return result

def colormap(img):
  img = np.uint8(img)
  result = mapping[img]
  result = np.reshape(result, (227,227,3))
  return result

legend = create_legend_image(mapping, dataset_labels)

def classify_image(img):
  img = cv2.resize(img, (227,227))
  in_ = np.array(img, dtype=np.float32)
  in_ = in_[:,:,::-1]
  in_ -= np.array((104.00698793,116.66876762,122.67891434))
  in_ = in_.transpose((2,0,1))
  # shape for input (data blob is N x C x H x W), set data
  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_
  # run net and take argmax for prediction
  net.forward()
  
  out_probs = net.blobs['prob'].data[...]
  out = out_probs[:,:].argmax(axis=1)
  seg = net.blobs['segmentations'].data[0]
  seg = seg.reshape(227,227)
  
  #Replace segId with prob argmax
  result = seg
  for j in range(0,out.shape[0]):
    result[seg == j] = out[j]
  result = result[:,:,np.newaxis]
  
  #cv2.imwrite('output.png',result)
  #result = cv2.applyColorMap(result, cv2.COLORMAP_JET);
  padding = 255 * np.ones((227,373,3))
  result = colormap(result)
  result = np.concatenate((padding,img,result,padding),axis=1)
  result = np.concatenate((result,legend),axis=0)
  return result


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    rc,img = capture.read()
                    if not rc:
                        continue
                    #result = img
                    result = classify_image(img)
                    #result = create_legend_image(mapping, dataset_labels)
                    #result = motion(img)
                    #result = face_detect(img)
                    #result = person_detect(img)
                    r, buf = cv2.imencode(".jpg",result)
                    self.wfile.write("--jpgboundary\r\n")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                    self.wfile.write('\r\n')
                    time.sleep(0.001)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html') or self.path=="/":
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="./cam.mjpg" />')
            self.wfile.write('</body></html>')
            return

def main():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 227); 
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 227);
    try:
        server = HTTPServer(('',8080),CamHandler)
        #server.socket = ssl.wrap_socket (server.socket, ca_certs='/home/pi/Authentication/ca.crt', certfile='/home/pi/Authentication/server.crt', keyfile='/home/pi/Authentication/server.nocrypt.key', server_side=False, cert_reqs=ssl.CERT_REQUIRED)
        print("server started")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()

