import cv2, os
import numpy as np
from PIL import Image

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        return (str(dir)+" created")
    return (str(dir)+" exist")

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    ids = []

    
    for imagePath in imagePaths:

        
        PIL_img = Image.open(imagePath).convert('L')

        img_numpy = np.array(PIL_img,'uint8')
        #print img_numpy

        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
        print ".",
    return faceSamples,ids

print "Getting Face data",
faces,ids = getImagesAndLabels('dataset')
print ""
print "Training.........."
recognizer.train(faces, np.array(ids))
print "TRAINED"

forTest=assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
print "Model saved"
print faces[0][1][0]


