#!/usr/bin/env python
# coding: utf-8

# # Need For Speed Gesture Play Controler
# 
# ### Demonstration on how to play https://youtu.be/u-FXfJzHhFQ?si=F-rfytD372bzSmHH
# 
# #### start the game. Pressing ALT+ENTER allows you to make most games display in a window.
# #### I like putting the game on full screen in windown and code on the right half of the screen
# #### Pressing ALT TAB allows you to switch between windows. 
# #### While the game is running run the code. A window will apear. it will also click on the right side of the screen hopefully selecting game
# #### to exit switch to the window produced by code and press x
# #### I would suggest sitting 3-4 feet from the camera 
# 
# ### Game controls
# #### There are two horisontal lines on the screen. 
# #### If both your hands are inbetween the lines nothing happens
# #### If your hsnds are above the top line the car will go forward. 
# #### If your hands are below the bottom line your pc will press back button to slow down and eventualy reverse. 
# #### To turn the wheel pretend that you are holding the wheel and turn. The code reads the angle between you hands. its displayed on the screen. if the number is small you will go straight if the angle increases the car will turn right. If the angle goes negative the car will turn left. 
# 
# 
# 
# 
# 

# ### Inspiration and tutorials used
# #### https://www.youtube.com/watch?v=06TE_U21FK4
# #### https://www.youtube.com/watch?v=We1uB79Ci-w

# ## Dowmload Free need for speed
# https://www.bestoldgames.net/need-for-speed-3-hot-pursuit

# 
# ## Preload libraries

# In[1]:


##!pip install pywinauto

##!pip install mediapipe

##!pip install opencv-python

##!pip install pyautogui


# In[6]:


import cv2
import numpy as np
import time
import mediapipe as mp
from win32api import GetSystemMetrics
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import pyautogui
from pywinauto.keyboard import send_keys, KeySequenceError
import win32api, win32con

import matplotlib.pyplot as plt
###https://github.com/pywinauto/pywinauto/issues/493
## default is 0.01

from pywinauto.timings import Timings
# for .type_keys(...)
Timings.after_sendkeys_key_wait = 0
cv2.destroyAllWindows()

import win32api, win32con
import math


# In[7]:


###https://github.com/pywinauto/pywinauto/issues/493
## default is 0.01

from pywinauto.timings import Timings
# for .type_keys(...)
Timings.after_sendkeys_key_wait = 0



# In[9]:


#NFS
#### setting up constants
font_size=1
fps=0
t0 = time.time()
width = int(GetSystemMetrics(0) )
scale_percent=.25
height =  int(GetSystemMetrics(1) )
dsize = (int(width*scale_percent), int(height*scale_percent))
winname = "NFS GesturePlay"  
cv2.namedWindow(winname) 
cv2.namedWindow(winname) 
cv2.moveWindow(winname, int((.5-scale_percent/2)*width),int((1-scale_percent*1.1)*height))
win32api.SetCursorPos((int(width *.8),int(height/2)))
pyautogui.mouseDown()
pyautogui.mouseUp()
color=(255, 255, 255)

neutral_zone=.6
break_zone=.8


###https://github.com/nicknochnack/MediaPipePoseEstimation/blob/main/Media%20Pipe%20Pose%20Tutorial.ipynb
###https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb
#### starting the loop
# define a video capture object
camera = cv2.VideoCapture(0)
n=0
xs=[]
names=[]
speed=10
press_time=.01
with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_tracking_confidence=0.7,
        min_detection_confidence=0.2) as pose:
 
    while(True):
        
        ### setting time
        tic = time.perf_counter()
        start=time.time()
        # Capture the video frame
        # by frame
    
        return_value, image = camera.read()
        image=np.real(camera.read()[1])
        image=cv2.flip(image, 1)
        cols,rows,channels = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            point=mp_pose.PoseLandmark.RIGHT_INDEX.value
            point_l=mp_pose.PoseLandmark.LEFT_INDEX.value
            visibility =results.pose_landmarks.landmark[point].visibility
            visibility_left =results.pose_landmarks.landmark[point_l].visibility
            dist=0
            if visibility>.7:
                x=round((results.pose_landmarks.landmark[point].x),2)
                y=round((results.pose_landmarks.landmark[point].y),2)
                hand =str([x,y])
            else: 
                x=0
                y=.95
                
                
                
            if visibility_left>.7:
                xl=round((results.pose_landmarks.landmark[point_l].x),2)
                yl=round((results.pose_landmarks.landmark[point_l].y),2)
                
                dist=yl-y
            else:   
                xl=0
                yl=.95
                
                
            hand_right=(int(xl*rows),int(yl*cols))
            hand_left=(int(x*rows),int(y*cols))
            
            mp_drawing.draw_landmarks( image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
            
            cv2.rectangle(image, (int(.2*rows), int(neutral_zone*cols)), (int(.8*rows), int(neutral_zone*cols)),(255, 255, 255), 2) 
            cv2.rectangle(image, (int(.2*rows), int(break_zone*cols)), (int(.8*rows), int(break_zone*cols)),(255, 255, 255), 2) 
            y_avg=(y+yl)/2

            
            cv2.putText(image, " : "+str(int(dist*100)),tuple([50, 240]), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 5, cv2.LINE_AA)
        except:
            pass 
        try:
               
            if (y<neutral_zone and yl<neutral_zone):
                    send_keys('{w down}', pause=0)
                    ###time.sleep(press_time)
            else:
                send_keys("{w up}", pause=0)        
            if(y>break_zone and yl>break_zone):
                send_keys('{s down}', pause=0)
            else:    ##time.sleep(press_time*10)
                send_keys("{s up}", pause=0)        
            if dist > .1:
                    send_keys('{d down}', pause=0)
            else:
                    send_keys("{d up}", pause=0)
            if dist<-.1:
                    send_keys('{a down}', pause=0)
            else:
                    send_keys("{a up}", pause=0)

                  
                
            
            ## storeing points
            points=results.pose_landmarks   
            
        except:
            pass 
            #predicting the action


        # Display the resulting frame
        cv2.putText(image, "fps:"+""+str(int(fps))+" d: "+str(int(dist*100)),(int(rows*0),int(cols*.1)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1, cv2.LINE_AA)
        cv2.line(image, hand_left, hand_right, color, 2) 
        image=cv2.resize(image, dsize)        
        cv2.imshow(winname, image)
        cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)

        t1 = time.time()
        timepass=t1-t0
        end=time.time()
        totalTime=end-start
        fps=1/totalTime   

        # the 'q' button is set as the
        # desired button of your choic
        
        if (cv2.waitKey(1) & 0xFF == ord('x')):
            break
    
    # After the loop release the cap objects
    camera.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




