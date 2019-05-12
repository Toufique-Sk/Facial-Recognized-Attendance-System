import cv2
import numpy as np
import os
import pymongo
from pymongo import MongoClient
import datetime
import pandas as pd

#connecting to database-- mongodb
client= MongoClient()
database= client.MarkOne

def getcurrentDateTime():
    datetime.datetime.now()
    return datetime.datetime.now()


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

filename=str(getcurrentDateTime().date())+".csv"

df=pd.read_csv(filename)

recognizer = cv2.face.LBPHFaceRecognizer_create()
name="Unknown"
assure_path_exists("trainer/")
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        dictt=database.RecogData.find({"_id": Id})
    
        for i in dictt:
            name=i['Name']
            department=i['Department']
            
        if confidence>40:
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, name, (x,y-40), font, 1, (255,255,255), 2)
            #cv2.putText(im, str(confidence), (x,y-100), font, 1, (255,255,255), 1)
            df.loc[df.ID==Id,'Exit-Time']=getcurrentDateTime().time()


            
        else:
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, "Unknown", (x,y-40), font, 1, (255,255,255), 2)
            #cv2.putText(im, "N/A", (x,y-100), font, 1, (255,255,255), 1)
        ##n=name+"/"
        ##assure_path_exists(n)
        ##os.startfile(name)
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

df.to_csv(filename)
