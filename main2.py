import streamlit as st
import json
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Charger les indices de classe à partir du fichier JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Charger le modèle entraîné
model = load_model('plant_disease_prediction_model.keras')

# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normaliser les valeurs de pixels
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension de lot
    return image

# Fonction pour prédire la classe de l'image
def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

st.image("herbre.png")
# Titre de l'application
st.markdown("<h1 style='text-align: center; color: green;'> Green Guardien </h1>", unsafe_allow_html=True)

# Logo de l'application dans la barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: green;'> Green Guardien </h1>", unsafe_allow_html=True)

st.sidebar.image('arbre.jpg', use_column_width=True)
st.sidebar.write("<h5 style='text-align: center;'>Votre Gardien Virtuel de la Santé des Plantes</h5>", unsafe_allow_html=True)
# Description de l'application
st.write(" <h5 style='text-align: center;'>Système de prédiction des maladies des plantes</h5><br><br>", unsafe_allow_html=True)

# Champ de saisie de texte pour le chemin de l'image avec une placeholder
image_path = st.text_input("Entrez le chemin de l'image :", value="", help="Entrez le chemin de l'image ici")

# Si un chemin d'image est entré
if image_path:
    try:
        # Charger et afficher l'image avec une taille de 150x150 pixels
        image = Image.open(image_path)
        resized_img = image.resize((150, 150))
        st.image(resized_img)

        # Bouton pour prédire la classe de l'image
        if st.button('Classify'):
            # Prédire la classe de l'image
            prediction = predict_image(image, model)
            st.success(f'Prediction: {str(prediction)}')
    except Exception as e:
        # Afficher un message d'erreur en cas d'erreur
        st.write("Erreur :", e)
