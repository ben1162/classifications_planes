import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_hub.keras_layer import KerasLayer
import plotly.graph_objects as px

from PIL import Image
import pickle
import sklearn
from sklearn.svm import SVC
import yaml




yaml_file = open("../app/app.yaml", 'r')
yaml_content = yaml.safe_load(yaml_file)

print("Key: Value")
for key, value in yaml_content.items():
    print(f"{key}: {value}")



IMAGE_WIDTH = yaml_content['IMAGE_WIDTH']
IMAGE_HEIGHT = yaml_content['IMAGE_WIDTH']
IMAGE_DEPTH = yaml_content['IMAGE_DEPTH']







def load_image(path):
    """Load an image as numpy array
    """
   # path = pathlib.Path('C:/Users/nathanael/Documents/Plane-classification/')
    return plt.imread(path)
    

    


def predict_image(Path_class,path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    
    names= pd.read_csv(Path_class,names=['Names'])
    image= [np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))]
    #print(image)
    prediction_vector = model.predict(np.array(image))
    #print(prediction_vector)
    predicted_classes = np.argmax(prediction_vector, axis=1)[0]
    names_classes=names['Names'][predicted_classes]
    #print(names)
    #print(predicted_classes)
    return prediction_vector, predicted_classes,names_classes

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)



bouton_ra = st.sidebar.radio(
     "Type of model",
     ('Reseaux de neurones', 'SVM', 'Transfert_learning'))
   
sidebar_model =  st.sidebar.selectbox(
                    'Target!',
                     ('Manufacturer', 'Family'))






st.title("Plane Identification ")

uploaded_file = st.file_uploader("Upload a plane image") #, accept_multiple_files=True)#



if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)
    
    
predict_btn = st.button("Identify", disabled=(uploaded_file is None and sidebar_model is None))

if predict_btn and sidebar_model =='Manufacturer'and bouton_ra == 'Reseaux de neurones':
    Path_class = yaml_content["DATA_DIR_CLASSE_MANUFACTURER"]
    model = load_model(yaml_content["DATA_DIR_MANIFACTURER"])
    model.summary()
    prediction_vector,prediction,classes = predict_image(Path_class,uploaded_file, model)
    #prediction='ben'
    st.title("Class")
    st.write(f"This is a : {classes}")
    st.title("Prediction")
    st.write(f"of Class number : {prediction}")
    st.title("Probability")
    st.write(f"with a probability : {prediction_vector}")
    st.title("Barchat")
    st.bar_chart(prediction_vector)
    
    
    
   #

elif predict_btn and sidebar_model =='Family'and bouton_ra == 'Reseaux de neurones':
    Path_class = yaml_content["DATA_DIR_CLASSE_FAMILY"]
    model = load_model(yaml_content["DATA_DIR_FAMILY"])
    prediction_vector,prediction,classes = predict_image(Path_class,uploaded_file, model)
    st.title("Class")
    st.write(f"This is a : {classes}")
    st.title("Prediction")
    st.write(f"of Class number : {prediction}")
    st.title("Probability")
    st.write(f"With a probability : {prediction_vector}")
    st.title("Barchat")
    st.bar_chart(prediction_vector)
    
    

    
elif predict_btn and sidebar_model =='Manufacturer' and bouton_ra == 'SVM':
    
    model = pickle.load(open(yaml_content["DATA_DIR_MANIFACTURER_SVM"],'rb'))
    Path_class = yaml_content["DATA_DIR_CLASSE_MANUFACTURER"]
    names = pd.read_csv(Path_class,names=['Names'])
    image = np.array(Image.open(uploaded_file).resize((IMAGE_WIDTH, IMAGE_HEIGHT))) / 255
    print(image)
    lenofimage = len(image)
    image_bon = image.reshape(1, -1)
    print(image_bon.shape)
    prediction = model.predict(image_bon)
    names_classes = names['Names'].astype('category').cat.categories[prediction]
    st.title("Class")
    st.write(f"This is a : {names_classes[0]}")
    st.title("Prediction")
    st.write(f"Of class number : {prediction[0]}")
    
        

     
    
elif predict_btn and sidebar_model =='Family' and bouton_ra == 'SVM':
    
    model = pickle.load(open(yaml_content["DATA_DIR_FAMILY_SVM"],'rb'))
    Path_class = yaml_content["DATA_DIR_CLASSE_FAMILY"]
    names = pd.read_csv(Path_class,names=['Names'])
    image = np.array(Image.open(uploaded_file).resize((IMAGE_WIDTH, IMAGE_HEIGHT))) / 255
    lenofimage = len(image)
    image_bon = image.reshape(1, -1)
    print(image_bon.shape)
    prediction = model.predict(image_bon)
    names_classes = names['Names'].astype('category').cat.categories[prediction]
    st.title("Class")
    st.write(f"this is a : {names_classes[0]}")
    st.title("Prediction")
    st.write(f"Of class number : {prediction[0]}")


    
  
    
    
    


elif predict_btn and sidebar_model =='Manufacturer' and bouton_ra == 'Transfert_learning':
    
    from tensorflow.keras.models import load_model
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = IMAGE_WIDTH
    IMAGE_DEPTH = 3
   
    model = load_model("manufacturer_tl.h5" , compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    
    Path_class = yaml_content["DATA_DIR_CLASSE_MANUFACTURER"]
    prediction_vector,prediction,classes = predict_image(Path_class,uploaded_file, model)
 
    st.title("Class")
    st.write(f"This is a : {classes}")
    st.title("Prediction")
    st.write(f"Of class number  : {prediction}")
    st.title("Probabilité")
    st.write(f"With the probability : {prediction_vector}")
    st.title("Barchat")
    st.bar_chart(prediction_vector)

    
    
elif predict_btn and sidebar_model =='Family' and bouton_ra == 'Transfert_learning':
    from tensorflow.keras.models import load_model
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = IMAGE_WIDTH
    IMAGE_DEPTH = 3
    model = load_model("family_tl.h5" , compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    Path_class = yaml_content["DATA_DIR_CLASSE_MANUFACTURER"]
    prediction_vector,prediction,classes = predict_image(Path_class,uploaded_file, model)
    st.title("Class")
    st.write(f"This is a : {classes}")
    st.title("Prediction")
    st.write(f"of class number : {prediction}")
    st.title("Probability")
    st.write(f"With a probability: {prediction_vector}")
    st.title("Barchat")
    st.bar_chart(prediction_vector)


else: print('Change the model')
    
