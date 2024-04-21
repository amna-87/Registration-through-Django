
from django.shortcuts import render, HttpResponse, redirect 
from django.contrib.auth.models import User 
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import numpy as np
import cv2
import random
import tensorflow as tf
from joblib import load
import librosa
import librosa.display
from typing import  Tuple
import os
import keras

# model_path = './models/saved-model.keras'
# # print("Absolute model path:", model_path)

# combined_network = keras.models.load_model(model_path)


# print("Absolute 2 model path:", model_path)

# Create your views here.
def preprocess_audio_series(raw_data: np.ndarray) -> np.ndarray:
    N, M = 24, 1319
    mfcc_data = librosa.feature.mfcc(y=raw_data, n_mfcc=24)
    mfcc_data_standardized = (mfcc_data - np.mean(mfcc_data)) / np.std(mfcc_data)
    number_of_columns_to_fill = M - mfcc_data_standardized.shape[1]
    padding = np.zeros((N, number_of_columns_to_fill))
    padded_data = np.hstack((padding, mfcc_data_standardized))

    # Reshaping to N,M,1
    return padded_data.reshape(N, M, 1)


def extractFrams(video_path):
    cap = cv2.VideoCapture(video_path)
    # Extract 6 frames from the video
    frames = []
    for i in range(6):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    # Close the video capture object
    cap.release()
    return frames

def resize_image(image: np.ndarray, new_size: Tuple[int,int]) -> np.ndarray:
    return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def crop_image_window(image: np.ndarray, training: bool = True) -> np.ndarray:
    height, width, _ = image.shape
    if training:
        MAX_N = height - 128
        MAX_M = width - 128
        rand_N_index, rand_M_index = random.randint(0, MAX_N) , random.randint(0, MAX_M)
        return image[rand_N_index:(rand_N_index+128),rand_M_index:(rand_M_index+128),:]
    else:
        N_index = (height - 128) // 2
        M_index = (width - 128) // 2
        return image[N_index:(N_index+128),M_index:(M_index+128),:]
    

def prediction():
    
    path = 'videos/testing'

    # frames =  extract_N_video_frames(file_path=video_path)

    frames=extractFrams(path)

    audio_data, sample_rate = librosa.load(path)

    preprocessed_audio = preprocess_audio_series(raw_data= audio_data.reshape(-1))

    print('==========\n'+path+'==========\n')

    resized_images = [resize_image(image= im, new_size= (128,1280)) for im in frames]

    cropped_images = [crop_image_window(image= resi,training= True) / 255.0 for resi in resized_images]

    preprocessed_video = np.stack(cropped_images)


    # ### Predict personality traits
    predicted_personality_traits = combined_network.predict([preprocessed_audio.reshape(1,24,1319,1),preprocessed_video.reshape(1,6,128,128,3)])

    # ### Print predicted personality traits
    personalities = ['Neuroticism','Extraversion','Agreeableness','Conscientiousness','Openness']
    for label, value in zip(personalities,predicted_personality_traits[0]):
        print(label + ': ' + str(value))

    # Load the video
    # display_frames(frames)

@login_required(login_url='login')
def HomePage(request):
    return render (request, 'home.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')
        
        if pass1!=pass2:
            return HttpResponse("your password doesnot match")
        else:
            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')
        
        
        print(uname,email,pass1, pass2)
    
    return render (request, 'signup.html')

def LoginPage(request):
    if request.method=='POST':
        
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse("username or password incorrect!")
            
        
    return render (request, 'login.html')


def LogoutPage(request):
    logout(request)
    return redirect('login')

def AboutPage(request):
    return render (request, 'about.html')



def TestPersonalityPage(request):
    prediction()
    return render (request, 'testPersonality.html')