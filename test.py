from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow


loaded_model = tensorflow.keras.models.load_model(
       ("my_model.h5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)


class_names = ["burger", "butter_naan", "chai","chapati","chole_bhature","dal_makhani","dhokla","fried_rice","idli","jalebi","kaathi_rolls","kadai_paneer","kulfi","masala_dosa","momos",
               "paani_puri","pakode","pav_bhaji","pizza","samosa"]

calories = {"burger":295 , "butter_naan":76, "chai":1,"chapati":71,"chole_bhature":500,"dal_makhani":278,"dhokla":152,"fried_rice":163,"idli":58,"jalebi":66,"kaathi_rolls":197,"kadai_paneer":248,"kulfi":184,"masala_dosa":539,"momos":280,"paani_puri":216,
            "pakode":315,"pav_bhaji":401,"pizza":266,"samosa":130}
def preprocess_input_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))

    image_array = np.asarray(image)

    normalized_image_array = image_array / 255.0

    input_data = np.expand_dims(normalized_image_array, axis=0)

    return input_data


image_path = 'chapati.jpg'
input_data = preprocess_input_image(image_path)

predictions = loaded_model.predict(input_data)
label = class_names[np.argmax(predictions)]

print(f"{calories[label]} calories are present in a serving of {label}.")


