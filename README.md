README for Face Recognition Code
This code uses a pre-trained Keras model for classifying images captured from a webcam. The goal of the program is to continuously capture frames from the webcam, preprocess them, and predict the class of the object or person in the frame, displaying the predicted class and its confidence score.

Requirements:
Python 3.x (Recommended version: 3.7 or higher)
Libraries:
Keras: Deep learning framework for building and training models.
TensorFlow: Backend for Keras (usually installed automatically with Keras).
OpenCV: Library for handling webcam inputs and image processing.
NumPy: Library for numerical operations, especially useful for manipulating arrays.
Install these libraries using pip:
Copy code
pip install tensorflow opencv-python numpy
Steps:
Pre-trained Model:

The code assumes you have a trained model file (keras_Model.h5). This model should be a Keras-based model, possibly a Convolutional Neural Network (CNN) for image classification.
Class Names:

The class labels are stored in a text file (labels.txt). Each line in the text file represents a different class name, corresponding to the possible outputs of the model.
Webcam Input:

The code uses OpenCV to capture images from your computer's webcam. It resizes each frame to 224x224 pixels to match the model's input size. The image is then normalized to fit the range expected by the model (values between -1 and 1).
Prediction:

The model predicts the class of the object or person in the frame. The predicted class is determined using np.argmax(), which picks the class with the highest confidence score.
The prediction and the confidence score are displayed in the terminal, where the confidence score is shown as a percentage.
Exit Condition:

The program runs in an infinite loop until the ESC key (ASCII value 27) is pressed. Once ESC is pressed, the webcam is released and the OpenCV window is closed.
Key Sections in the Code:
Model Loading:

The model is loaded using load_model(). Make sure the model file keras_Model.h5 is in the same directory as the script or provide the full path.
Webcam Image Capture:

The frame is captured using camera.read() and resized to fit the model's input requirements using cv2.resize().
Preprocessing:

The captured frame is converted into a NumPy array and reshaped to the format that the model expects. The pixel values are normalized by dividing by 127.5 and subtracting 1.
Prediction:

The model's prediction is stored in prediction, which is then used to find the index of the class with the highest probability (np.argmax()).
Displaying Results:

The predicted class name and its confidence score are printed on the terminal, showing only the percentage part of the score (rounded).
Exit Mechanism:

The loop continues to run until the ESC key is pressed. Upon exit, the webcam is released and all OpenCV windows are closed.
Troubleshooting:
Webcam Access Issue: Ensure that no other application is using the webcam when running this script.
Model and Labels: Ensure the keras_Model.h5 file and labels.txt file are correctly located and contain the expected content. Labels must be listed in the same order as the model's output classes.
Example labels.txt:
python
Copy code
Person 1
Person 2
Person 3
Dog
Cat
...
Conclusion:
This code provides a basic framework for using a webcam for object or face recognition with Keras. By tweaking the model and labels, you can adapt this to various real-time classification tasks.

Here’s a structured step-by-step guide for building a Face Recognition System using a model trained with Google Teachable Machine and integrating it with Python:

1. Collect and Prepare Data
Use Google Teachable Machine to collect images for different individuals.
Capture images from various angles, lighting conditions, and expressions for better accuracy.
Export the trained model in TensorFlow Lite or Keras format.
2. Set Up the Python Environment
Install necessary Python libraries:
bash
Copy code
pip install opencv-python tensorflow numpy
3. Load the Trained Model in Python
Use TensorFlow to load the exported model.
Example:
python
Copy code
import tensorflow as tf

model = tf.keras.models.load_model('path_to_model')
4. Capture Real-Time Video Input
Use OpenCV to access the webcam.
python
Copy code
import cv2

cap = cv2.VideoCapture(0)  # Access webcam
while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
5. Preprocess Captured Frames
Resize and normalize the frames as per the model’s input requirements.
python
Copy code
import numpy as np

def preprocess(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input
    normalized_frame = resized_frame / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
6. Perform Face Detection (Optional but Recommended)
Use a face detector to focus on faces only.
python
Copy code
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces
7. Run Face Recognition Prediction
Predict the identity of detected faces.
python
Copy code
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
"C:\Users\91935\OneDrive\Pictures\Screenshots\Screenshot 2025-01-07 212613.png"

8. Optimize and Test the System
Test with different lighting and angles.
Fine-tune by collecting more training data if necessary.
9. (Optional) Deploy the Model
Convert the model to TensorFlow Lite for mobile deployment.
Deploy to Raspberry Pi for edge devices.
"C:\Users\91935\OneDrive\Pictures\Screenshots\Screenshot 2025-01-07 212253.png"




