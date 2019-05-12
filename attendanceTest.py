import unittest
import pandas as pd
import cv2
import numpy as np
import pymongo
from pymongo import MongoClient

single_empdet=[]
id_from_csv=[]

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual(id_from_csv,single_empdet )


if __name__ == '__main__':
    #connecting to database-- mongodb
    client= MongoClient()
    database= client.MarkOne


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

   

    font = cv2.FONT_HERSHEY_SIMPLEX

    im=cv2.imread("testImage.jpeg")

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
            if Id not in single_empdet:
                    single_empdet.append(Id)

        else:
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, "Unknown", (x,y-40), font, 1, (255,255,255), 2)



    cv2.imshow('image',im)

    if cv2.waitKey(0) == 27:        
        cv2.destroyAllWindows()


    df=pd.read_csv("2019-05-11.csv")
    
    for i in df['ID']:
        id_from_csv.append(i)
    unittest.main()