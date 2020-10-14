from tkinter import *
import cv2
from PIL import Image,ImageTk
import imutils
import os
import numpy as np
import face_recognition
import pickle
from imutils import paths

import _thread

def compute_iou(boxA,boxB):
    
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    
    interArea=max(0,xB-xA+1)*max(0,yB-yA+1)
    
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    
    iou=interArea/float(boxAArea+boxBArea-interArea)
    return iou


class MY_GUI():

    def __init__(self,init_window_name):
        
        self.init_window_name=init_window_name

    def set_init_window(self):
        self.init_window_name.title('人脸识别demo')
        #self.init_window_name.geometry('600x600')  
        #self.init_window_name["bg"] = "pink"
        #self.init_window_name.attributes("-alpha",0.9)

        self.init_window_name.bind("<KeyPress>", self.call_back)

        self.v1 = StringVar()
        self.v2=StringVar()

        self.frame=Frame(self.init_window_name)
        self.frame.pack(side=TOP,fill=BOTH,expand=True)
        self.read_data=Button(self.frame, height=2,text='拍摄人像', command=self.camera,background = "#003366", fg = "white")
        self.read_data.pack(side=LEFT,expand=True,fill=BOTH,padx=10)

        self.stop_camera=Button(self.frame, height=2,text='停止拍摄', command=self.stop,background = "#003366", fg = "white")
        self.stop_camera.pack(side=LEFT,expand=True,fill=BOTH,padx=10)

        self.save_data=Button(self.frame,height=2,text='识别测试',command=self.recognition,background = "#006699", fg = "white")
        self.save_data.pack(side=LEFT,expand=True,fill=BOTH,padx=10)

        self.stop_re=Button(self.frame, height=2,text='停止识别', command=self.stop2,background = "#003366", fg = "white")
        self.stop_re.pack(side=LEFT,expand=True,fill=BOTH,padx=10)


        self.frame2=Frame(self.init_window_name)
        self.frame2.pack(side=TOP,fill=BOTH,expand=True,pady=10)
        self.L1 = Label(self.frame2, text="请输入姓名") 
        self.L1.pack(side = LEFT,expand=True,fill=BOTH,padx=10)
        self.name_entry = Entry(self.frame2,textvariable=self.v1,bd=3,show=None)
        self.name_entry.pack(side=LEFT,expand=True,fill=BOTH,padx=10)

        self.frame3=Frame(self.init_window_name)
        self.frame3.pack(side=TOP,fill=BOTH,expand=True,pady=10)
        self.L2 = Label(self.frame3, text="请输入姓名拼音") 
        self.L2.pack(side = LEFT,expand=True,fill=BOTH,padx=10)
        self.name_entry2 =Entry(self.frame3,textvariable=self.v2,bd=3,show=None)
        self.name_entry2.pack(side=LEFT,expand=True,fill=BOTH,padx=10)

        self.label=Label(self.init_window_name,text='welcome to face-recognition demo')
        self.label.pack(side=TOP,expand=True,fill=BOTH,padx=10,pady=10)
        self.panel = Label(self.init_window_name,height=300,width=400)  # initialize image panel
        self.panel.pack(padx=10, pady=10)
        self.detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cam=cv2.VideoCapture(0)

        self.ok=1
        self.re=0
        self.getnew=0
        self.total=0
        self.k=0

    def call_back(self,event):

        if event.keysym=='k':
            self.k=1

        else:
            self.k=0

    def encode_img(self):

        if self.getnew:
            dataset='data_test/'+self.name_entry2.get()
            imagePaths=list(paths.list_images(dataset))
            knownEncodings=[]
            knownNames=[]   
            for (i,imagePath) in enumerate(imagePaths):
                self.label['text']='INFO processing image {}/{}'.format(i+1,len(imagePaths))
                name=imagePath.split(os.path.sep)[-2]
                image=cv2.imread(imagePath)
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
                boxes=face_recognition.face_locations(rgb,model='hog')
                encodings=face_recognition.face_encodings(rgb,boxes)
        
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(self.name_entry.get())

            filename='info.pickle'
            if os.path.isfile(filename):
                data=pickle.loads(open(filename,'rb').read())
                data['encodings'].extend(knownEncodings)
                data['name'].extend(knownNames)
            else:
                data={}
                data['encodings']=knownEncodings
                data['name']=knownNames

            f=open(filename,'wb')
            f.write(pickle.dumps(data))
            f.close()
            self.getnew=0
        self.label['text']='finished'

    
    def stop(self):
        self.ok=0
        _thread.start_new_thread(self.encode_img,())
        self.name_entry['state']='normal'
        self.name_entry2['state']='normal'

    def camera(self):
    
        if len(self.name_entry.get())<2 or len(self.name_entry2.get())<2 or not self.name_entry2.get().isalpha():
            self.ok=0
            self.label['text']='请输入姓名'
        else:
            
            self.name_entry['state']='disabled'
            self.name_entry2['state']='disabled'
            self.label['text']='press "K" to get pic'
            self.ok=1
            self.path=os.path.sep.join(['data_test',self.name_entry2.get()])
            if not os.path.exists('data_test'):
                os.mkdir('data_test')
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            self.path=self.name_entry2.get()
            _thread.start_new_thread(self.get_pic,())

    def get_pic(self,):

        if self.ok:

            ret, frame = self.cam.read()  # 从摄像头读取照片
            if ret!=None:

                orig=frame.copy()
                frame=cv2.resize(frame,(300,400))
                
                rects=self.detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
                for(x,y,w,h) in rects:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.putText(frame,'get '+str(self.total)+' pictures',(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,255),2)
        
                if self.k==1:
                    p= os.path.sep.join(['data_test',self.path, "{}.png".format(str(self.total).zfill(5))])
                    self.total+=1
                    cv2.imwrite(p,orig)
                    self.k=0
                    self.getnew=1

                                
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                imgtk = ImageTk.PhotoImage(image=current_image)

                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
                self.init_window_name.after(1,self.get_pic)

        else:
            img_zero = np.zeros((400,400,3), np.uint8)
            cv2image=cv2.cvtColor(img_zero,cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(cv2image)#将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)


    def re_process(self):

        if self.re:

            ret, frame = self.cam.read()  # 从摄像头读取照片
            if ret!=None:

                orig=frame.copy()
                frame=cv2.resize(frame,(400,400))
                rects=self.detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
                
                
                frame=cv2.rectangle(frame,(100,150),(250,300),(255,0,0),2)

                if self.count==0 and len(rects)!=0:
                
                    for(x,y,w,h) in rects:

                        iou=compute_iou([100,150,250,300],[x,y,x+w,y+h])
      
                        if iou>0.6:

                                print(iou)

                                self.count=1

                                self.frame_c=frame[100:300,100:300,:]
                   
                                _thread.start_new_thread(self.re_staff,())


                                break  
                        else:

                            self.label['text']='请对准方形区域'
                                                        
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                imgtk = ImageTk.PhotoImage(image=current_image)


                
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
                self.init_window_name.after(1,self.re_process)


    def re_staff(self):

        filename='info.pickle'

        data=pickle.loads(open(filename,'rb').read())


        rgb=cv2.cvtColor(self.frame_c,cv2.COLOR_BGR2RGB)
        
        boxes=face_recognition.face_locations(rgb,model='hog')
        print(boxes)
        encodings=face_recognition.face_encodings(rgb,boxes)
        names=[]
        for encoding in encodings:
            matches=face_recognition.compare_faces(data['encodings'],encoding,tolerance=0.39)
            name='Unknown'
    
            if True in matches:
                
                matchedIdxs=[i for (i,b) in enumerate(matches) if b]
                counts={}
                for i in matchedIdxs:
                    name=data['name'][i]
                    counts[name]=counts.get(name,0)+1
                
                name = max(counts,key=counts.get)
            names.append(name)

        if len(names)==0:
            self.count=0
        else:
            for ((top,right,bottom,left),name) in zip(boxes,names):
        
                cv2.rectangle(self.frame_c,(left,top),(right,bottom),(255,0,0),2)
            
            self.label['text']=str(names)

            self.re=1

       

    def recognition(self):

        self.re=1
        self.count=0

        _thread.start_new_thread(self.re_process,())



    def stop2(self):
        self.re=0
        img_zero = np.zeros((300,400,3), np.uint8)
        cv2image=cv2.cvtColor(img_zero,cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.label['text']='finished'



def gui_start():

    init_window = Tk()              #实例化出一个父窗口

    ZMJ_PORTAL = MY_GUI(init_window)

    ZMJ_PORTAL.set_init_window()


    init_window.mainloop()   


gui_start()