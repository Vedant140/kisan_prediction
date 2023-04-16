import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

Corn=['Blight','Common Rust','Gray Leaf Spot','Healthy']
Wheat=['Fusarium Head Blight','Leaf Rust','Tan Spot']
Rice=['Bacterial Leaf Blight','Blast','Brownspot']
Cotton=['Diseased Leaf','Diseased Plant','Fresh Leaf','Fresh Plant']
Potato=['Early Blight','Late Blight','']

st.title("Disease prediction")
objects = ['Wheat', 'Rice', 'Corn','Cotton','Potato']
selected_object = st.selectbox('Select the crop:', objects)
st.write('You selected:', selected_object)
if selected_object == 'Wheat':
    model_path = 'wheat_vedant.h5'
elif selected_object == 'Rice':
    model_path = 'rice_vedant.h5'
elif selected_object == 'Corn':
    model_path = 'corn_vedant.h5'
elif selected_object == 'Potato':
    model_path = 'Potato.h5'
else:
    model_path = 'Cotton.h5'

model = tf.keras.models.load_model(model_path)
def make_prediction(model, input_image):
    img_array = np.array(input_image.convert('RGB').resize((256, 256)))
    img_array = img_array / 255.
    prediction = model.predict(np.array([img_array]))
    return prediction

uploaded_file = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        prediction = make_prediction(model, image)

if selected_object=='Corn':
    a=np.argmax(prediction)
    if a==1:
        st.write(Corn[0])
        with open('CornDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==2:
        st.write(Corn[1])
        with open('CornDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[1]
        st.write(selected_page_content)
    elif a==3:
        st.write(Corn[2])
        with open('CornDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[2]
        st.write(selected_page_content)
    else:
        st.write('Healthy plant')

if selected_object=='Wheat':
    a=np.argmax(prediction)
    if a==0:
        st.write(Wheat[0])
        with open('WheatDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==1:
        st.write(Wheat[1])
        with open('WheatDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[1]
        st.write(selected_page_content)
    elif a==2:
        st.write(Wheat[2])
        with open('WheatDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[2]
        st.write(selected_page_content)

if selected_object=='Rice':
    a=np.argmax(prediction)
    print(a)
    if a==0:
        st.write(Rice[0])
        with open('RiceDiseaseInfo _1_.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==1:
        st.write(Rice[1])
        with open('RiceDiseaseInfo _1_.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==2:
        st.write(Rice[2])
        with open('RiceDiseaseInfo _1_.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)

if selected_object=='Cotton':
    a=np.argmax(prediction)
    if a==0:
        st.write(Cotton[0])
        with open('CottonDisease.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==1:
        st.write(Cotton[1])
        with open('CottonDisease.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==2:
        st.write(Cotton[2])
        with open('CottonDisease.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    else:
        st.write(Cotton[3])
        with open('CottonDisease.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)

if selected_object=='Potato':
    a=np.argmax(prediction)
    if a==0:
        st.write(Corn[0])
        with open('PotatoDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==1:
        st.write(Corn[1])
        with open('PotatoDiseaseInfo.txt', 'r') as f:
            file_contents = f.read()
        pages = file_contents.split('\n\n')
        selected_page_content = pages[0]
        st.write(selected_page_content)
    elif a==2:
        st.write("Healthy Plant")
