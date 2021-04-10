import cv2
import mediapipe as mp   #for hand_detection
import time


class handDetector():

    # default constructor
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):  # parameters
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # getting hand model data in mpHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        # draw hand points on img
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        #get the color image from the img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # do process on selected img
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)   #landmark {x: 0.72558576 y: 0.7405052 z: 0.08141178}

        if self.results.multi_hand_landmarks:
            #select each landmark/poits on hands
            for handlms in self.results.multi_hand_landmarks:

                # if draw is true it will draw hand coonections b/w points
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

        # at the end it will return img with points and connection
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            # get lanmark/points data and store in myHands
            myHand = self.results.multi_hand_landmarks[handNo]
            #print(myHand)

            for id, lm in enumerate(myHand.landmark):  #fetch the data from myhands with id and location of point
                #print(id,lm)
                #return id and location of all points (id,x,y,z)

                h, w, c = img.shape  #return imge size (480,640,3)

                cx, cy = int(lm.x * w), int(lm.y * h)       #to get exact location of points in the img
                lmList.append([id, cx, cy])                 #store the values in list

                #drwa the circle on the given points
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

                #print(lmList)

            return lmList


def main():
    pTime = 0

    cap = cv2.VideoCapture(0) #to get img
    detector = handDetector() #obj of class

    while True:
        success, img = cap.read()
        img = detector.findHands(img)    #calling method of class findHands to detect the hand

        lmList = detector.findPosition(img)  #calling method of class findPosition to detect the landmarks of hand points
        #it will return list of all the landmarks with Id and pisitions of landmarks


        # if lmList != None:   # if list is not empty then only it will print the list
        #     print(lmList)


        #to clalculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        #img presentation
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()