import numpy as np
import tensorflow as tf
import cv2
import sys
import os
import time

os.chdir(os.getcwd())

# face detector && pad set-up
faced_dir = './haarcascade_frontalface_alt.xml'
export_dir = './lib'
faceCascade = cv2.CascadeClassifier(faced_dir)

# Basic model parameters.
IMAGE_SIZE = 256 #input image size

# name_scope
inputname = "input:0"
outputname = "Mean_2:0"#SecondAmin/


def face_score(image):

    with tf.Session() as sess:
    # load the facepad model
        tf.saved_model.loader.load(sess, 
    			[tf.saved_model.tag_constants.SERVING], 
    			export_dir)
        _input = tf.get_default_graph().get_tensor_by_name(inputname)
        _output = tf.get_default_graph().get_tensor_by_name(outputname)
        score = sess.run(_output,feed_dict={_input : image})

    return(score)


def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img,1.3,5)

    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

scores=[]
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(300,300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        

        
        file_name_path = 'inputs/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        
        img=cv2.imread(file_name_path)
        
        scores.append(face_score(img))
        
       
        #cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    if cv2.waitKey(1)==13 or count==10:
    	print(sum(scores)/len(scores))
    	break
    

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!!!')