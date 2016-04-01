#!/usr/bin/python
'''
	orig author: Igor Maculan - n3wtron@gmail.com
  edited by Steven Hickson
	A Simple mjpg stream http server
'''
import cv2
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import time
from threading import Thread
import numpy as np

capture=None
result=None

def img_thread():
    global result
    while True:
        try:
            rc,img = capture.read()
            if not rc:
                continue
            result = img
            time.sleep(0.01)
        except KeyboardInterrupt:
            break
    return

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print self.path
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
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
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640); 
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480);
    try:
        thread = Thread(target = img_thread)
        thread.daemon = True
        thread.start()
        server = HTTPServer(('',8080),CamHandler)
        print "server started"
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()

