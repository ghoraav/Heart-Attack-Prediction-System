import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io


#streamlit run ecg_analyzer.py


# Load the trained model
@st.cache_resource  # This will cache the model loading
def load_trained_model():
    try:
        model = tf.keras.models.load_model('ecg_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image Preprocessing (same as training)
def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to same size as training
    image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values
    image = image / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    return image

def main():
    st.title("ECG Heart Attack Risk Analyzer")
    st.write("Upload an ECG image to analyze potential heart attack risks")
    
    # Load the trained model
    model = load_trained_model()
    
    if model is None:
        st.error("Please follow these steps to train the model first:")
        st.code("""
        1. Prepare your dataset in folders (normal/abnormal)
        2. Run the training script:
           python train_ecg_model.py
        3. Make sure the generated 'ecg_model.h5' is in the same folder as this app
        """)
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded ECG', use_column_width=True)
        
        if st.button('Analyze ECG'):
            with st.spinner('Analyzing ECG...'):
                # Convert to numpy array and preprocess
                image_array = np.array(image)
                processed_image = preprocess_image(image_array)
                
                # Make prediction using loaded model
                prediction = model.predict(np.expand_dims(processed_image, axis=0))[0][0]
                
                # Display results
                st.subheader('Analysis Results')
                
                # Risk level based on prediction
                if prediction > 0.5:
                    risk_level = "High"
                    color = "red"
                elif prediction < 0.5:
                    risk_level = "Low"
                    color = "Green"
                
                
                # Display risk level with color
                st.markdown(f"### Risk Level: :${color}[{risk_level}]")
                st.write(f"Prediction Score: {prediction:.2f}")
                st.write(f"Risk Level: 0.0-0.5 - Low Risk, 0.5-1.0 - High Risk")
                
                # Detailed analysis
                st.subheader('Detailed Analysis')
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Risk Factors:")
                    if prediction > 0.5:
                        st.write("- Significant ECG abnormalities detected")
                        st.write("- Pattern suggests elevated risk")
                        
                    elif prediction < 0.5:
                        st.write("- No significant abnormalities detected")
                        st.write("- ECG patterns appear normal")
                        
                        
                
                with col2:
                    st.write("Recommendations:")
                    if prediction > 0.5:
                        st.write("- Consult a healthcare provider")
                        st.write("- Further medical evaluation recommended")
                    else:
                        st.write("- Continue regular check-ups")
                        st.write("- Maintain heart-healthy lifestyle")
                

if __name__ == "__main__":
    main()