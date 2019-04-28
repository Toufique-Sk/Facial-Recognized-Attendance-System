import cv2
import os
import pymongo
from pymongo import MongoClient

#creating and connecting database-- mongodb
client= MongoClient()
database= client.SampleDatabase

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#directory checking and creating
def assure_path_exists(path):
    dir = os.path.dirname(path)
    print (dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

def count_no_of_image():
    path, dirs, files = next(os.walk("dataset/"))
    file_count = len(files)
    return file_count

print ("Enter Department Name:")
dep_Name=raw_input()
print ("Enter Roll No.: ")
face_id=input()
print ("Enter Name: ")
name=raw_input()


dictid={"_id":face_id, "Name":name, "Department": dep_Name}
database.RecogData.insert_one(dictid)

count = count_no_of_image()
count1=count

vid_cam = cv2.VideoCapture(0)
assure_path_exists("dataset/")

while(True):

    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        count += 1

        cv2.imwrite("dataset/User."+str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif (int(count)-count1)>=200:
        break
vid_cam.release()
cv2.destroyAllWindows()
