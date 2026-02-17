# Heart Attack Prediction System 🫀

An advanced Artificial Intelligence system designed to assess and predict the risk of heart attacks. This project employs a dual-stream approach: analyzing **numerical health metrics** through Machine Learning and **ECG images** using Deep Learning to provide a comprehensive cardiovascular risk assessment.

---

## 🌟 Key Features
* **Hybrid Data Analysis:** Processes both clinical numerical data and ECG image files.
* **Deep Learning Integration:** Utilizes a Convolutional Neural Network (CNN) for automated ECG classification.
* **Machine Learning Ensemble:** Implements XGBoost, Gradient Boosting, and Logistic Regression for high-precision clinical predictions.
* **Interactive UI:** Built with a Streamlit-based web interface for easy patient data entry and image uploading.
* **Risk Recommendations:** Provides personalized medical suggestions based on high/low risk scores.

---

## 🧪 Methodologies

### 1. Numerical Analysis (Clinical Data)
The system evaluates patient health profiles based on 11 key parameters, including Age, Sex, Chest Pain Type, Blood Pressure, and Cholesterol.

**Models Implemented:**
* **XGBoost:** Optimized for high-performance clinical decision-making and error correction.
* **Artificial Neural Network (ANN):** A multi-layered model built with Keras to capture complex, non-linear relationships in health data.
* **Logistic Regression & SGD:** Reliable baseline classifiers used for binary risk outcomes.



### 2. ECG Image Analysis (Deep Learning)
A Convolutional Neural Network (CNN) architecture is trained to distinguish between "Normal" and "Abnormal" ECG patterns to identify early warning signs of cardiac distress.

* **Preprocessing:** Automated grayscale conversion and resizing to 224x224 pixels using OpenCV.
* **Data Augmentation:** Applied rotation, zoom, and shifting to mimic real-world variances and improve model robustness.
* **Architecture & Optimization:** Utilizes Batch Normalization, Max Pooling, and the Adam optimizer with Binary Cross-Entropy loss.



[Image of a Convolutional Neural Network architecture for image classification]


---

## 🛠 Tech Stack
* **Software:** Python, Visual Studio Code, Jupyter Notebook.
* **Libraries:** TensorFlow, Keras, OpenCV, Scikit-learn, XGBoost, Pandas, NumPy.
* **Deployment:** Streamlit.

---

## 📊 Results
* **High-Risk Detection:** Successfully identifies critical cases (e.g., advanced age combined with hypertension) and prompts immediate medical consultation.
* **Low-Risk Detection:** Accurately filters healthy patterns, recommending preventive measures and a heart-healthy lifestyle.
