from django.shortcuts import render,HttpResponse,redirect
from json import dumps
from HumanActionIdentification.settings import BASE_DIR
from .forms import Video_form
from .models import Video
from .load import inputCoordinates
from .apps import HaiConfig
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
# Create your views here.
def index(request):
    all_video=Video.objects.all()
    #print("inn")
    if request.method == "POST":
        #print("in if")
        form=Video_form(data=request.POST,files=request.FILES)
        
        if form.is_valid():
            print("in from if")
            form.save()
            return HttpResponse("<h1> Uploaded successfully </h1>")
        else:
            print(request.POST)
            print(request.FILES)
            print(form.non_field_errors)
            print(form.errors)
    else:
        #print("in else")
        form=Video_form()
    return render(request,'index.html',{"form":form,"all":all_video})

def home(request):
    return render(request,'home.html')

def videos(request):
    all_video=Video.objects.all()
    
    return render(request,'video.html',{"all":all_video})

def predict(request):
    list1=[]
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    cap = cv2.VideoCapture(0)
    #print(cv2.getBuildInformation())
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #print("In cap is opened")
        while cap.isOpened():
            ret, frame = cap.read()
            #print("In cap is opened")
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                row = pose_row
                
    #             # Append class name 
    #             row.insert(0, class_name)
                
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 

                # Make Detections
                #print('hello')
                X = pd.DataFrame([row])
                body_language_class = HaiConfig.model.predict(X)[0]
                body_language_prob = HaiConfig.model.predict_proba(X)[0]
                #print(body_language_class, body_language_prob)
                list1.append(inputCoordinates(body_language_class,body_language_prob))
                #print('after')
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                
                
            except Exception as e:
                #print("exeption : ",e)
                pass
                            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return render(request,'predResult.html',{'list':list1})

def predictUploaded(request):
    list1=[]
    #lastvideo= Video.objects.last()
     
    val1=str(request.POST["sel"])
    #print("sel : ",val1)
    p='\\Users\\admin\\Desktop\\BEproj\\HumanActionIdentification'+val1
    path=os.path.join(os.path.join(BASE_DIR,p))#os.path.dirname(__file__),'Input_Check.mp4')
    #print(path)
    
    #print(lastvideo.video.url)
    #print(os.path.join(BASE_DIR,lastvideo.video.url))
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    cap = cv2.VideoCapture(path)#lastvideo.video.url)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #print("in with")
        while cap.isOpened():
            #print("in isopend")
            ret, frame = cap.read()
            if ret==True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    row = pose_row
                    
        #             # Append class name 
        #             row.insert(0, class_name)
                    
        #             # Export to CSV
        #             with open('coords.csv', mode='a', newline='') as f:
        #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #                 csv_writer.writerow(row) 

                    # Make Detections
                    #print('hello')
                    X = pd.DataFrame([row])
                    body_language_class = HaiConfig.model.predict(X)[0]
                    body_language_prob = HaiConfig.model.predict_proba(X)[0]
                    #print(body_language_class, body_language_prob)
                    list1.append(inputCoordinates(body_language_class,body_language_prob))
                    #print('after')
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    
                    
                except Exception as e:
                    #print("exeption : ",e)
                    pass
                                
                cv2.imshow('Raw Webcam Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    return render(request,'predResult.html',{'list':list1})

def help(request):
    return render(request,'help.html')

def delete(request):
    Video.objects.all().delete()
    return render(request,'video.html')
    