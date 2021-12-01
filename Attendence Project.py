#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
 
import cv2 
import numpy as np 
import face_recognition as fr
import os
from datetime import datetime as dt


# In[ ]:


path = "Images"
clsnames = []
imgs = []
mylist = os.listdir(path)
#print(mylist)

for cls in mylist:
    currentImg = cv2.imread(f'{path}/{cls}')
    imgs.append(currentImg)
    clsnames.append(os.path.splitext(cls)[0])
    
#print(clsnames)


def find_face_encodings(images):
    encode_list = []
    for cur_img in images:
        cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2RGB)
        cur_img_encode = fr.face_encodings(cur_img)[0]
        encode_list.append(cur_img_encode)
    return encode_list

def mark_attendance(name):
    with open("attendance.csv",'r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = dt.now()
            TimeString = now.strftime("%H:%M:%S")
            DateString = now.strftime("%d-%m-%Y")
            f.write(f'\n{name},{TimeString},{DateString}')


encode_list_known = find_face_encodings(imgs)    
#print(len(encode_list_known))
print("encoding completed successfully")

name_matched=""
capture_img = cv2.VideoCapture(0)

while True:
    success, img2 = capture_img.read()
    images2 = cv2.resize(img2,(0,0),None,0.25,0.25)
    images2 = cv2.cvtColor(images2,cv2.COLOR_BGR2RGB)
    face_locs = fr.face_locations(images2)
    face_encode = fr.face_encodings(images2,face_locs)
    
    
    for encode_face,faceLoc in zip(face_encode,face_locs):
        matches = fr.compare_faces(encode_list_known,encode_face)
        face_distance = fr.face_distance(encode_list_known,encode_face)
        #print(face_distance)
        match_face_index = np.argmin(face_distance)

        if matches[match_face_index]:
            name = clsnames[match_face_index].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img2,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img2,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img2,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            mark_attendance(name)
           
    cv2.imshow('webcam',img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture_img.release()
cv2.destroyAllWindows()

