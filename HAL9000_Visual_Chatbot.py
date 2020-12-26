''' THIS IS THE SPYDER CODE FOR RUNNING IN STREAMLIT '''

import numpy as np 
import os
import io
import datetime
from gtts import gTTS 
import webbrowser
from io import BytesIO


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
 
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
    return href

def classify():
    st.image('https://t3.ftcdn.net/jpg/00/96/55/94/240_F_96559467_Fxgsa20HIuPGWywzEDnBMy3NokapCzxH.jpg',width=420)
    st.write("Hi! May I help you with the place?")
    #ph=st.text_input("Enter the image path .... ")
    model = load_model("AISUCCESS3_with_new_train.h5")
	
    """if st.checkbox('Select a file in current directory'):
        folder_path = '.'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)"""
	
    #uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    #ph="Bekal_Fort9.jpg"
	

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    im = Image.open(uploaded_file)
    a = np.asarray(im)
    result = Image.fromarray(a)
    filename = get_image_download_link(result)
    st.markdown(filename, unsafe_allow_html=True)
	
    img = load_img(filename, target_size=(227,227))
  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    classes = model.predict_classes(img, batch_size=10)
    spot={0:'Agra Fort',1:'Ajanta and Ellora Caves',2:'Amer Fort',3:'Bangalore Palace',4:'Basilica of Bom Jesus',5:'Bekal Fort',6:'Charminar',7:'City Palace',8:'Elephanta Cave',9:'Fatehpur Sikri',10:'Gateway of India',11:'Gingee Fort',12:'Golden Temple',13:'Golkonda Fort',14:'Gwalior Fort',15:'Hawa Mahal',16:'Hill Palace',17:'Howrah Bridge',18:'Humayuns Tomb',19:'India Gate',20:'Jama Masjid',21:'Janta Mantir',22:'Kaye Monastry',23:'Konark Sun Temple',24:'Lotus Temple',25:'Madurai Meenakshi Temple',26:'Mysore Palace',27:'Nalanda University',28:'Qutub Minar',29:'Ran ki Vav',30:'Rashtrapati Bhavan',31:'Red Fort',32:'Sanchi Stupa',33:'Shore Temple Mahabalipuram',34:'Taj Mahal',35:'Thanjavur Chola Temple',36:'Victoria Memorial',37:'Victoria Terminal',38:'Vidhana Soudha',39:'Vivekananda Rock Memorial'}

    place=spot[classes[0]]
    if filename is not None:
    	image = Image.open(filename)
    	#buf = io.BytesIO()
    	#image.save(buf, format='JPEG')
    	#byte_im = buf.getvalue()
    	st.image(image, caption='Uploaded Image.', use_column_width=True)
    	st.write(" ")
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
            st.text_area("Bot:", value="Thank You, Have a nice day!", height=200, max_chars=None, key=None)
            break
        result = predictor.predict(passage=passage,question=inp)
        response = result['best_span_str']
        st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)
        
        mytext = response
        myobj = gTTS(text=mytext, lang=language, slow=False)
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        filename = "voice"+date_string+".ogg"
        myobj.save(filename)
        name = "Text Data/"+filename
        audio_file = open(name, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
        os.remove(name)
        
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
            name = "Text Data/"+location+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()
            chat(passage)
        else:
            st.write(" I'm really sorry for that.")
            labels=['Agra Fort','Ajanta and Ellora Caves','Amer Fort','Bangalore Palace','Basilica of Bom Jesus','Bekal Fort','Charminar','City Palace','Elephanta Cave','Fatehpur Sikri','Gateway of India','Gingee Fort','Golden Temple','Golkonda Fort','Gwalior Fort','Hawa Mahal','Hill Palace','Howrah Bridge','Humayuns Tomb','India Gate','Jama Masjid','Janta Mantir','Kaye Monastry','Konark Sun Temple','Lotus Temple','Madurai Meenakshi Temple','Mysore Palace','Nalanda University','Qutub Minar','Ran ki Vav','Rashtrapati Bhavan','Red Fort','Sanchi Stupa','Shore Temple Mahabalipuram','Taj Mahal','Thanjavur Chola Temple','Victoria Memorial','Victoria Terminal','Vidhana Soudha','Vivekananda Rock Memorial']
            tour= st.selectbox('Please Choose the location to continue chatting', labels )
            name = "Text Data/"+tour+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()
            chat(passage)
             
    elif choice =="About":
        webbrowser.open_new_tab("about.html")
         
if __name__ == '__main__':
	main()        
