#!/usr/bin/python
'''
	orig author: Igor Maculan - n3wtron@gmail.com
	A Simple mjpg stream http server
'''
import cv2
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import time
import ssl

capture=None
imgN1=None
imgN2=None
face_cascade=None
hog=None

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def motion(img):
    global imgN1
    global imgN2
    if imgN1 is not None and imgN2 is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        result = diffImg(imgN2, imgN1, img)
        #    result = img
    else:
        result = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgN2 = imgN1
    imgN1 = img
    return result

def face_detect(img):
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    bwImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(bwImg, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
    return img

def person_detect(img):
    global hog
    if hog is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    return img

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
                    result = img
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
                    time.sleep(0.05)
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

