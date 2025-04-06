import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model(r'C:\Users\prash\Downloads\Final\Final\model.h5')

# Disease categories
categories = ['early blight', 'late blight', 'healthy']

# ---------- CSS Styling with Background ----------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e0f7fa, #e8f5e9);
            background-attachment: fixed;
        }

        .main-container {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(6px);
            padding: 40px 30px;
            border-radius: 20px;
            box-shadow: 0px 4px 30px rgba(0,0,0,0.1);
            margin: auto;
            max-width: 800px;
        }

        h1 {
            color: #2e7d32;
            text-align: center;
            font-size: 2.5rem;
        }

        .uploaded-image {
            text-align: center;
            margin-top: 20px;
        }

        .prediction {
            font-size: 24px;
            color: #1b5e20;
            font-weight: bold;
            text-align: center;
            margin-top: 30px;
        }

        .confidence-alert {
            font-weight: bold;
            color: #d84315;
            text-align: center;
            margin-top: 20px;
        }

        .footer {
            text-align: center;
            font-size: 13px;
            color: #5f6368;
            margin-top: 60px;
        }

        .stButton>button {
            background-color: #388e3c;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 10px 24px;
            margin-top: 10px;
        }

        .stFileUploader label,
        .stSelectbox label {
            font-weight: bold;
            color: #2e7d32;
        }

        .sidebar .sidebar-content {
            background-color: #c8e6c9;
        }

        .css-1d391kg {
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Potato_Solanum_tuberosum.jpg/320px-Potato_Solanum_tuberosum.jpg", use_column_width=True)
    st.header("🌿 About")
    st.markdown("""
    This is a deep learning-powered tool that detects **potato leaf diseases**.

    - Upload a clear image of a leaf.
    - Only one leaf per image.
    - Supported formats: `.jpg`, `.png`, `.jpeg`.

    _Model: Trained using TensorFlow on leaf disease dataset._
    """)
    st.markdown("📬 [Contact Developer](mailto:prashanthbandi987@email.com)")

# ---------- Main Layout ----------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf and get an AI prediction of its condition.")

# Optional dropdown
st.selectbox("Optional: Choose a disease (for comparison)", categories)

# Upload image
uploaded_image = st.file_uploader("Upload Potato Leaf Image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
    st.image(uploaded_image, caption="Uploaded Image", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)

    confidence_scores = predictions[0]
    predicted_class = categories[np.argmax(confidence_scores)]
    confidence = np.max(confidence_scores)

    # Confidence chart
    st.subheader("📊 Confidence Scores")
    fig, ax = plt.subplots()
    bar_colors = ['#ff8a80', '#4db6ac', '#81c784']
    ax.bar(categories, confidence_scores, color=bar_colors)
    ax.set_ylabel("Confidence")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Prediction result
    st.markdown(f'<div class="prediction">Prediction: {predicted_class} <br>Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)

    # Warning if low confidence
    if confidence < 0.6:
        st.markdown('<div class="confidence-alert">⚠️ Low confidence – consider uploading a higher quality image.</div>', unsafe_allow_html=True)

    # Expand to show all scores
    with st.expander("🔍 View detailed prediction scores"):
        for i, score in enumerate(confidence_scores):
            st.write(f"- {categories[i]}: {score*100:.2f}%")

# ---------- Footer ----------
st.markdown("""
    <div class="footer">
        <hr style="border: none; height: 1px; background-color: #ddd; margin: 30px 0;">
        <h4 style="color:#2e7d32;">👨‍💼 Developed By</h4>
        <ul style="list-style-type: none; padding-left: 0; font-size: 15px;">
            <li>✨ Devendra Reddy ✨ Ramanji ✨ Chandra Reddy ✨ Prashanth ✨ Nithin Reddy</li>
        </ul>
        <p style="margin-top:20px;">Built with ❤️ using Streamlit & TensorFlow<br>
        &copy; 2025 Potato Leaf Disease Classifier</p>
    </div>
</div>
""", unsafe_allow_html=True)
