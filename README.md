# Sports Recommendation System

This is a Flask-based sports recommendation system that suggests sports activities, time duration, and weekly frequency based on user preferences and characteristics. The system uses a pre-trained TensorFlow model to make predictions.

## Prerequisites

- Python 3.9
- Flask
- TensorFlow 2.15.0
- NumPy
- Gunicorn
- scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sports-recommendation-system.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sports-recommendation-system
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure you have the following files in the project directory:
   - `sports_rec.h5`: The pre-trained TensorFlow model.
   - `mlb_mot.pkl`: The pickled MultiLabelBinarizer object for encoding motivation features.
   - `mlb_health.pkl`: The pickled MultiLabelBinarizer object for encoding health condition features.

2. Run the Flask application:

   ```bash
   python app.py
   ```

   The application will start running on `http://localhost:8080`.

3. Send a POST request to the `/predict` endpoint with the following JSON payload:

   ```json
   {
     "age": 25,
     "gender": "male",
     "placePreference": "dalam",
     "socialPreference": "sendiri",
     "equipment": "sedang",
     "availableTime": "30-45 menit",
     "fitnessLevel": "Average",
     "motivation": ["Meningkatkan Konsentrasi dan Fokus"],
     "diseaseHistory": ["physical injury"]
   }
   ```

   Replace the values in the JSON payload with the appropriate user preferences and characteristics.

4. The API will respond with the recommended sports activities, time duration, and weekly frequency:

   ```json
   {
     "sportsRecommendations": [
       "Badminton/Tennis (olahraga raket)",
       "Cycling",
       "Walking/Jogging/Running"
     ],
     "timeRecommendations": "30-45 menit",
     "weeklyRecommendations": "2x"
   }
   ```

## Deployment

The application can be deployed using Docker and Cloud Build. The provided `Dockerfile` and `cloudbuild.yaml` files can be used for deployment.

1. Build the Docker image:

   ```bash
   docker build -t sports-recommendation-system .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 8080:8080 sports-recommendation-system
   ```

3. Use the `cloudbuild.yaml` file to set up continuous deployment with Cloud Build.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The pre-trained TensorFlow model and pickled MultiLabelBinarizer objects were trained and generated separately.
- The Flask framework is used for building the web application.
- Gunicorn is used as the WSGI server for production deployment.
