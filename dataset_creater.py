import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0); #0 for internal camera and 1 for external camera

def insertorupdate(id,Name,age):
    conn=sqlite3.connect("sqlite.db")
    cmd="SELECT * FROM Students WHERE ID="+str(id)
    cursor=conn.execute(cmd)  #cursor is a pointer to execute the statement
    isRecordExist=0    #to check if the record is already present or no record is present
    for row in cursor:
        isRecordExist=1;
    if(isRecordExist==1):
        conn.execute("UPDATE STUDENTS SET Name=? WHERE Id=?", (Name,id))
        conn.execute("UPDATE STUDENTS SET Age=? WHERE Id=?", (age,id))
    else: #if the record is not present then insert the record
        conn.execute("INSERT INTO STUDENTS(id,Name,age) VALUES(?,?,?)",(id,Name,age))


    conn.commit()
    conn.close()


#insert user defined values into table
id=input('enter user Id: ')
name=input('enter Name: ')
age=input('enter Age: ')


insertorupdate(id,name,age)

#detect face in web camera coding

sampleNum=0;
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Image convert into BGRGRAY Color
    faces=faceDetect.detectMultiScale(gray,1.3,5); #Scale faces
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1; #if face is detected then increment the sample number
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100); #delay time
    cv2.imshow("Face",img); #show faces detected in web camera
    cv2.waitKey(1);
    if(sampleNum>20):
        break;

cam.release()
cv2.destroyAllWindows()




