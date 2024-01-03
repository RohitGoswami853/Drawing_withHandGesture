import cv2 
import mediapipe as mp
import time
import numpy as np

video=cv2.VideoCapture(0)
video.set(3,720)
video.set(4,1280)
imgcanvas=np.zeros((480,720,3),np.uint8)

mpHands=mp.solutions.hands
hand=mpHands.Hands(min_detection_confidence=0.85)
mpDraw=mp.solutions.drawing_utils
ptime=0
xp,yp=0,0
while True:
    _,img=video.read()
    img=cv2.flip(img,1)
    img = cv2.resize(img, (720, 480)) 
    imgRBG=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hand.process(imgRBG)
    lmlist=[]
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if id==8 :
                    cv2.circle(img,(cx,cy),10,(123,33,8),cv2.FILLED)
                #print(lmlist)

            # Drawing:
            x1,y1=lmlist[8][1:]
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if lmlist[8][2]<lmlist[12][2]:
                cv2.line(img,(xp,yp),(x1,y1),(123,33,8),thickness=3)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),(123,33,8),thickness=5)
                print("DRAWING MOD")
                xp,yp=x1,y1
            elif lmlist[8][2]<lmlist[6][2] and lmlist[12][2]<lmlist[10][2] and lmlist[16][2]<lmlist[14][2] and lmlist[20][2]<lmlist[18][2]:
                print("ERASE MOD")
                cv2.line(img,(xp,yp),(x1,y1),(255,255,255),thickness=100)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),(0,0,0),thickness=100)
                xp,yp=x1,y1
            elif lmlist[8][2]>lmlist[12][2]:
                print("HOVER MODE")
           

            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
            
    
    
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    f=f"Current FPS: {int(fps)}"
    #print(f)
    img=cv2.addWeighted(img,0.2,imgcanvas,0.8,0)
    cv2.imshow("IMG",img)
    #cv2.imshow("CANVAS",imgcanvas)
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break


