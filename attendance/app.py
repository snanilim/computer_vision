import tkinter as tk
from tkinter import Message, Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='#effffb')

window.geometry("1500x800")

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#path = "profile.jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
#panel = tk.Label(window, image = img)


#panel.pack(side = "left", fill = "y", expand = "no")

#cv_img = cv2.imread("img541.jpg")
#x, y, no_channels = cv_img.shape
#canvas = tk.Canvas(window, width = x, height =y)
#canvas.pack(side="left")
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img)) 
# Add a PhotoImage to the Canvas
#canvas.create_image(0, 0, image=photo, anchor=tk.NW)

#msg = Message(window, text='Hello, world!')

# Font is a tuple of (font_family, size_in_points, style_modifier_string)



message = tk.Label(window, text="Intelligent Attendance Management System", bg="#effffb", fg="#617be3"  ,width=50  ,height=3,font=('times', 30, 'italic', 'underline')) 

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="#617be3"  ,bg="#d3f4ff" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20  ,bg="#d3f4ff" ,fg="#617be3",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="#617be3"  ,bg="#d3f4ff"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="#d3f4ff"  ,fg="#617be3",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="#617be3"  ,bg="#d3f4ff"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="#d3f4ff"  ,fg="#617be3"  ,width=30  ,height=2, activebackground = "#d3f4ff" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="#617be3"  ,bg="#d3f4ff"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="red"   ,bg="#d3f4ff",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def is_number(userId):
    if userId.isdigit():
        return True
    else:
        return False


def TakeImages():
    userId = (txt.get())
    userName = (txt2.get())
    # if is_number(userId):
    if True:
        cam = cv2.VideoCapture(2)
        cascadeFacePath = './cascades/data/haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascadeFacePath)
        camExitNum = 0
        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for(x,y,w,h) in faces:
                cv2.rectangle(gray, (x,y), (x+w, y+h), (255,0,0),2)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                camExitNum += 1
                cv2.imwrite("saveTrainImage/ "+userName +"."+userId +'.'+ str(camExitNum) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('frame', gray)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif camExitNum == 60:
                break 
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + userId +" Name : "+ userName
        row = [userId, userName]
        with open('studentDetails/studentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(userId)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(userName.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cascadeFacePath = './cascades/data/haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascadeFacePath)
    faces, userIDs = getImageLabels('saveTrainImage')
    # print(np.array(userIDs))
    recognizer.train(faces, np.array(userIDs))
    recognizer.save('trainingImageLabel/trainner.yml')
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)


def getImageLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    
    faces = []
    userIDs = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L') #loading the image and converting it to gray scale
        imageNP = np.array(pilImage, 'uint8')
        userId = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNP)
        userIDs.append(userId)
        # print(imageNP)
    return faces, userIDs

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainingImageLabel/trainner.yml")

    cascadeFacePath = './cascades/data/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadeFacePath)

    df = pd.read_csv("studentDetails/studentDetails.csv")

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            print(conf)
            if(conf<50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt = aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = "Unknown"
                tt = str(Id)

            cv2.putText(frame, str(tt), (x,y+h), font, 1, (255, 255, 255), 2)
            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="attendance/attendance_"+date+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)






clearButton = tk.Button(window, text="Clear", command=clear, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="red"  ,bg="#b2dffb"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Sheba AI Team","",)
copyWrite.configure(state="disabled",fg="#617be3")
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)
 
window.mainloop()