# PRODIGY_ML_04

## Hand Gesture Recognition with Web Interface (Flask)

This project demonstrates a hand gesture recognition system using a deep learning model. The system is accessible through a web interface built with Flask, allowing users to upload images and receive predictions about the gesture depicted.

***Features***
  
  Made three different version but in last version:
  - Hand gesture recognition using a convolutional neural network (CNN).
  - Web interface for uploading images and displaying results.
  - Responsive design using Bootstrap for a better user experience.

***Dataset***

The dataset used consists of near-infrared images of 10 different hand gestures, captured by the Leap Motion sensor. The images are organized into folders based on the subject identifier and gesture type.

***Project Structure***
  - *A.py:* Main Flask application script.
  - *V1/V2/V3:* Script to define and train the CNN model.
  - *templates/upload.html:* HTML template for the image upload page.
  - *templates/result.html:* HTML template for displaying the prediction results.
  - *static/uploads/:* Directory to store uploaded images.
  - *BModel/best_model/hand_gesture_model.h5:* Pre-trained model file.

***Requirements***
  - Python 3.x
  - Flask
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Pillow (PIL)
  You can install the required packages using pip:
    - pip install flask tensorflow keras numpy pandas pillow

***Getting Started***
  1. Clone the repository
      - git clone https://github.com/shitbaKashif/PRODIGY_ML_04
      - cd hand-gesture-recognition
  
  3. Download the dataset
      - Download the hand gesture dataset and organize it as described in the project structure. Download from: https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data?select=leapGestRecog.
  
  4. Train the Model (Optional)
      - If you want to train the model from scratch, run the jupiter notebooks:
      - This will save the trained model to best_model.h5.
  
  5. Start the Flask Application
      - Run the Flask application:
        - python A.py
      - Open a web browser and navigate to http://127.0.0.1:5000/ to access the web interface.

***Usage***
  - Upload an Image
  - On the upload page (http://127.0.0.1:5000/), click the "Choose File" button and select an image of a hand gesture.
  - Click the "Upload" button to submit the image.
  - View the Prediction
  - After uploading the image, you will be redirected to the result page, which displays the uploaded image and the predicted gesture.
