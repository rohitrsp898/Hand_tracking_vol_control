import numpy as np
import cv2
import time

#using handdetector model
import Handtrackingmodel as htm
import math

#for controlling system audio

#pip install pycaw                                 #//   https://github.com/AndreMiras/pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#initialization

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


#volume.GetMute()
#volume.GetMasterVolumeLevel()
#volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)


#get volume range
volRange=volume.GetVolumeRange()   #// (-65.25, 0.0, 0.03125)

minVol =volRange[0]        #   -65.25
maxVol =volRange[1]        #    0.0

#initial values
volbar=400
volper=0

#webcam window size
wCam,hCam=650,450

#webcam input
cap=cv2.VideoCapture(0)
#webcam window size
cap.set(3,wCam)
cap.set(4,hCam)

pTime=0

#calling handdetector class from the handtrackingmodel

detector=htm.handDetector(detectionCon=0.7)  #set min_hand_confidence to 70%

while True:

    success, img= cap.read()

    # calling method of class findHands to detect the hand
    img= detector.findHands(img)
    # calling method of class findPosition to detect the landmarks of hand points
    lmList=detector.findPosition(img,draw=False) #stores the position in lmList with id

    if lmList!=None: #list must have some values then only next step should excute
        #print(lmList[4],lmList[8])

        x1,y1=lmList[4][1], lmList[4][2]        #thumb top point location (x1,y1)
        x2,y2=lmList[8][1], lmList[8][2]        #first finger top point location (x1,y1)

        cx,cy=(x1+x2)//2,(y1+y2)//2             #mid point btw first finger(8) and thumb(4)

        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)           #draw circle on thumb point
        cv2.circle(img, (x2, y2), 10, (255, 0, 255),cv2.FILLED)     #draw circle on forst finger point
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)     #draw circle on mid point btw thumb and first finger

        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)                 #draw line btw thumb and first finger

        length=math.hypot(x2-x1,y2-y1)                              #calculate lenght btw thumb and first finger
        #print(length)       #30-200

        vol=np.interp(length,[15,150],[minVol,maxVol])              #convert the length value to volRange (15=-65.25, 150=0.0)
        #print(int(length),vol)

        volume.SetMasterVolumeLevel(vol, None)                      # set the system value as per the input-vol (15=-65.25 minvol, 150=0.0 maxval)

        volbar = np.interp(length, [15, 150], [400, 140])           #convert the lenght value to volbar (15 lenght=400 volbar-> min, 150 lenght=140 volbar -> max)
        volper = np.interp(length, [15, 150], [0, 100])             #convert the lenght value to volper (15 lenght=0 volper-> min, 150 lenght=100 volper -> max)
        #print(length,volbar)

        if length<20:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)      #change themid circle color   like button effect

    #volBar
    cv2.rectangle(img, (40, 140), (15, 400), (0, 255, 0),3)                     #draw empty volbar ractangle  inital position =400
    cv2.rectangle(img, (40, int(volbar)), (15, 400), (0, 255, 0),cv2.FILLED)    #draw filled volbar rectangle  initial position= 400

    #volPer
    cv2.putText(img,f'{int(volper)} %',(30,110),cv2.FONT_HERSHEY_PLAIN,4,(0,255,0),3) #draw volper of img

    #to calculate FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}',(45,45),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)


    cv2.imshow("Image",img)
    cv2.waitKey(1)

