from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('sports_rec.h5')

# Load the multilabel binarizer encoding
with open("mlb_mot.pkl", "rb") as file:
    mlb_mot = pickle.load(file)

with open("mlb_health.pkl", "rb") as file:
    mlb_health = pickle.load(file)

# Define the mappings used for encoding the features
age_mapping = {"15 to 18": 0, "19 to 25": 1, "26 to 30": 2, "31 to 39": 3, "40 and above": 4}
gender_mapping = {"female": 0, "male": 1}
location_mapping = {"dalam": 0, "luar": 1}
preferensi_mapping = {"sendiri": 0, "dengan orang lain": 1}
equipment_mapping = {"murah": 0, "sedang": 1, "mahal": 2}
waktu_mapping = {"<15 menit": 0, "15-30 menit": 1, "30-45 menit": 2, "45-60 menit": 3, ">60 menit": 4}
levelfitness_mapping = {"Unfit": 0, "Average": 1, "Good": 2}

# Define the label mappings
label_mapping = {
    0: "Sepakbola/Futsal/Minisoccer",
    1: "Badminton/Tennis (olahraga raket)",
    2: "Basket",
    3: "Voli",
    4: "Renang",
    5: "Cycling",
    6: "Senam/Aerobik",
    7: "Walking/Jogging/Running",
    8: "Gym",
    9: "Pilates/Yoga",
    10: "Dance/Zumba",
    11: "Billiard",
    12: "Softball/Kasti/Cricket",
    13: "Hiking",
    14: "Golf"
}

waktu_label_mapping = {
    0: "<15 menit",
    1: "15-30 menit",
    2: "30-45 menit",
    3: "45-60 menit",
    4: ">60 menit"
}

weekly_label_mapping = {
    0: "1x",
    1: "2x"
}

# Define the endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    # Preprocess the input data
    motivation = mlb_mot.transform([data["motivation"]])[0]
    healthconditions = mlb_health.transform([data["health_conditions"]])[0]
    age = age_mapping[data['age']]
    gender = gender_mapping[data['gender']]
    location = location_mapping[data['location']]
    preferensi = preferensi_mapping[data['preferensi']]
    equipment = equipment_mapping[data['equipment']]
    waktu = waktu_mapping[data['waktu']]
    levelfitness = levelfitness_mapping[data['levelfitness']]

    # Create the input features array
    sports_input = np.array([np.concatenate([motivation, healthconditions, [age], [gender], [location], [preferensi], [equipment]])]) 
    waktu_input = np.array([[waktu]])
    weekly_input = np.array([[levelfitness]])

    # Make predictions using the loaded model
    sports_pred, waktu_pred, weekly_pred = model.predict({'sports_input': sports_input,
                                                          'waktu_input': waktu_input,
                                                          'weekly_input': weekly_input})

    # Get the predicted class labels
    sports_labels = np.argsort(sports_pred)[-3:][::-1]
    waktu_label = np.argmax(waktu_pred)
    weekly_label = 1 if weekly_pred > 0.5 else 0
    
    # Map the predicted labels to their corresponding values
    sports_recommendations = [label_mapping[label] for label in sports_labels]
    waktu_recommendation = waktu_label_mapping[waktu_label]
    weekly_recommendation = weekly_label_mapping[weekly_label]

    # Prepare the response
    response = {
        'sports_recommendations': sports_recommendations,
        'waktu_recommendation': waktu_recommendation,
        'weekly_recommendation': weekly_recommendation
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)