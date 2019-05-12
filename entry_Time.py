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




recognizer = cv2.face.LBPHFaceRecognizer_create()
name="Unknown"
assure_path_exists("trainer/")
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
single_empdet=[]
emp_details=[]
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
            if Id not in single_empdet:
                single_empdet.append(Id)
                single_empdet.append(name)
                single_empdet.append(department)
                single_empdet.append(getcurrentDateTime().time())
                single_empdet.append("NULL")
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

## Create dataFrame
count=0
temp=[]
fullData=[]
for i in single_empdet:
    count+=1
    temp.append(i)
    if(count==5):
        count=0
        fullData.append(temp)
        temp=[]

# Create the pandas DataFrame 
df = pd.DataFrame(fullData, columns = ['ID', 'Name','Department','Entry-Time','Exit-Time']) 
df.set_index('ID',inplace=True)
datee=getcurrentDateTime().date()
filename=str(datee)+".csv"
df.to_csv(filename)
