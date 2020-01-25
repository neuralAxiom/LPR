import time
import numpy as  np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys, os
import keras
import cv2
import traceback
from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
import os 
import numpy as np
import sys
import copy
from src.label				import dknet_label_conversion
from src.utils 				import nms

import signal,sys,time                          
import objectDetection

cwd = os.getcwd()
print(cwd)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def signal_handling(signum,frame):           
    global terminate                         
    terminate = True    

wpod_net = load_model("lp-detector/wpod-net_update1.h5")
lp_threshold = .5


if __name__ == "__main__":

    img = cv2.imread('test1.jpg')
    img_darknet = img[:, :, ::-1]
    rgbImage = img_darknet.astype(np.uint8)
    results = objectDetection.getbBox(img_darknet)
    for i in results:
       x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
       xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
       pt1 = (xmin, ymin)
       pt2 = (xmax, ymax)
       print(xmin, ymin, xmax, ymax)
       cv2.rectangle(rgbImage, pt1, pt2, (0, 255, 0), 2) #.astype(np.uint8)
       cv2.putText(rgbImage, str(i[0]),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0)) #.astype(np.uint8)
       Ivehicle = img_darknet[ymin:ymax, xmin:xmax, :]
       ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
       side  = int(ratio*288.)
       bound_dim = min(side + (side%(2**4)),608)
       Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
       if len(LlpImgs):
          Ilp = LlpImgs[0]
          Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
          Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

          s = Shape(Llp[0].pts)
          
          cv2.imwrite('test_lp.png' ,Ilp*255.)
          imglp = (Ilp*255.).astype(np.uint8) 
          R = objectDetection.getOcr(imglp)
          print(R, imglp.shape)
          if len(R):

             L = dknet_label_conversion(R,imglp.shape[1],imglp.shape[0])
             L = nms(L,.45)

             L.sort(key=lambda x: x.tl()[0])
             lp_str = ''.join([chr(l.cl()) for l in L])

#             with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
#                f.write(lp_str + '\n')

             print '\t\tLP: %s' % lp_str

          else:
              print 'No characters found'
#          writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

#    cv2.imshow("img", Ilp)
#    cv2.waitKey(0)

