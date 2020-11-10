"Direct path given | Image printed as bytes | Speech | About | YesorNo"
import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import random
import os
#from pathlib import Path
#import tensorflow as tf
import io
import datetime
from gtts import gTTS 
import webbrowser

# Language in which you want to convert 
language = 'en'
     
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
 
def classify():
    st.write("Hi! May I help you with the place?")
    ph=st.text_input("Enter the image path .... ")
    model = load_model("C:\\Users\\91965\\Documents\\BDA SEM III\\MOM\\MOM AI Project\\TryInHP\\AISUCCESS3_with_new_train.h5")
    img = load_img(ph, target_size=(227,227))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    classes = model.predict_classes(img, batch_size=10)
    spot={0:'Agra Fort',1:'Ajanta and Ellora Caves',2:'Amer Fort',3:'Bangalore Palace',4:'Basilica of Bom Jesus',5:'Bekal Fort',6:'Charminar',7:'City Palace',8:'Elephanta Cave',9:'Fatehpur Sikri',10:'Gateway of India',11:'Gingee Fort',12:'Golden Temple',13:'Golkonda Fort',14:'Gwalior Fort',15:'Hawa Mahal',16:'Hill Palace',17:'Howrah Bridge',18:'Humayuns Tomb',19:'India Gate',20:'Jama Masjid',21:'Janta Mantir',22:'Kaye Monastry',23:'Konark Sun Temple',24:'Lotus Temple',25:'Madurai Meenakshi Temple',26:'Mysore Palace',27:'Nalanda University',28:'Qutub Minar',29:'Ran ki Vav',30:'Rashtrapati Bhavan',31:'Red Fort',32:'Sanchi Stupa',33:'Shore Temple Mahabalipuram',34:'Taj Mahal',35:'Thanjavur Chola Temple',36:'Victoria Memorial',37:'Victoria Terminal',38:'Vidhana Soudha',39:'Vivekananda Rock Memorial'}

    place=spot[classes[0]]
    if ph is not None:
        image = Image.open(ph)
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        st.image(byte_im, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        st.write("Classifying . . . . . . . . . . . . . . .")
        st.write("Classified")
        
        st.subheader('You are at %s ' % (spot[classes[0]]))
    return place

def chat(passage):
    st.write("Excited to know more about the place?!")
    st.write("Here's HAL9000 to help you with your queries!")
    st.write("**type _bye_ to end the chat")
    while True:
        inp = st.text_input("You: ", key='1')
        if inp.lower() == "bye":
            break
        result = predictor.predict(passage=passage,question=inp)
        response = result['best_span_str']
        st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)
        
        mytext = response
        myobj = gTTS(text=mytext, lang=language, slow=False)
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        filename = "voice"+date_string+".ogg"
        myobj.save(filename)
        name = "C:\\Users\\91965\\Documents\\BDA SEM III\\MOM\\MOM AI Project\\TryInHP\\"+filename
        audio_file = open(name, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
        os.remove(name)
        
#Width: 945
#Height: 788
            

def main():
    
    menu = ["Chatbot","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Chatbot":
        st.title("HAL9000")
        st.subheader(" Welcome to your visual chatbot assistant HAL9000 v.2.1.0. Over Over! \n Let's Predict ")
        location=classify()
        
        x=["Yes","No"]
        st.write(" Did I Get it Right?")
        choose=st.selectbox("Choose",x)
        if choose == "Yes":
            name = "C:\\Users\\91965\\Documents\\BDA SEM III\\MOM\\MOM AI Project\\TryInHP\\Text Data\\"+location+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()#.encode("utf-8")
            chat(passage)
        else:
            st.write(" I'm really sorry for that.")
            labels=['Agra Fort','Ajanta and Ellora Caves','Amer Fort','Bangalore Palace','Basilica of Bom Jesus','Bekal Fort','Charminar','City Palace','Elephanta Cave','Fatehpur Sikri','Gateway of India','Gingee Fort','Golden Temple','Golkonda Fort','Gwalior Fort','Hawa Mahal','Hill Palace','Howrah Bridge','Humayuns Tomb','India Gate','Jama Masjid','Janta Mantir','Kaye Monastry','Konark Sun Temple','Lotus Temple','Madurai Meenakshi Temple','Mysore Palace','Nalanda University','Qutub Minar','Ran ki Vav','Rashtrapati Bhavan','Red Fort','Sanchi Stupa','Shore Temple Mahabalipuram','Taj Mahal','Thanjavur Chola Temple','Victoria Memorial','Victoria Terminal','Vidhana Soudha','Vivekananda Rock Memorial']
            tour= st.selectbox('Please Choose the location to continue chatting', labels )
            name = "C:\\Users\\91965\\Documents\\BDA SEM III\\MOM\\MOM AI Project\\TryInHP\\Text Data\\"+tour+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()
            chat(passage)
            
    elif choice =="About":
        webbrowser.open_new_tab("C:\\Users\\91965\\Documents\\BDA SEM III\\MOM\\MOM AI Project\\TryInHP\\about.html")
         
if __name__ == '__main__':
	main()



        

        
        
        
        
