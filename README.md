# 🎭 Emotion Tracker (Real-Time Facial Emotion Detection)

A real-time emotion detection system built using **OpenCV, TensorFlow,
and Streamlit**.\
This project detects human faces from a webcam feed and classifies
emotions such as **Happy, Sad, Angry, Fear, Surprise, Neutral, and
Disgust**.

------------------------------------------------------------------------

## 🚀 Features

-   📸 Real-time face detection using OpenCV\
-   🧠 Emotion classification using a trained CNN model\
-   🎥 Live webcam integration\
-   🌐 Streamlit web app interface\
-   📓 Training notebook included (Google Colab)\
-   ⚡ Lightweight and easy to run locally

------------------------------------------------------------------------

## 🧠 Tech Stack

-   Python\
-   TensorFlow / Keras\
-   OpenCV\
-   NumPy\
-   Streamlit

------------------------------------------------------------------------
## 📂 Project Structure

```
emotion-tracker/
│
├── model/
│   └── emotion_model.keras
│
├── src/
│   └── inference.py
│
├── expression (1).ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 📊 Model Details

-   Dataset: FER-2013\
-   Input Size: 48x48 grayscale images\
-   Classes: 7 emotions\
-   Model: Convolutional Neural Network (CNN)\
-   Accuracy: \~54% (can be improved with tuning)

------------------------------------------------------------------------

## 📓 Model Training (Important)

The model was trained using Google Colab.

👉 To retrain or understand the training pipeline: - Open
`expression (1).ipynb`\
- Upload it to Google Colab\
- Run all cells step-by-step

------------------------------------------------------------------------

## ⚙️ Installation & Setup

``` bash
git clone https://github.com/your-username/emotion-tracker.git
cd emotion-tracker

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Project

``` bash
python src/inference.py
```

or

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🧪 How It Works

1.  Webcam captures live video\
2.  OpenCV detects faces\
3.  Face is resized to 48x48 grayscale\
4.  CNN predicts emotion\
5.  Result is displayed

------------------------------------------------------------------------

## 📈 Future Improvements

-   Improve model accuracy\
-   Add confidence scores\
-   Add analytics dashboard\
-   Deploy online

------------------------------------------------------------------------

## ⭐ If you like this project

Give it a star on GitHub!
