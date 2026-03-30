import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINE MODEL ARCHITECTURE (64-3-3) ---
class LetterNN(nn.Module):
    def __init__(self):
        super(LetterNN, self).__init__()
        self.hidden = nn.Linear(64, 3) 
        self.output = nn.Linear(3, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.output(x)

# --- 2. LOAD TRAINED MODEL ---
@st.cache_resource
def load_model():
    model = LetterNN()
    try:
        model.load_state_dict(torch.load('model.pth'))
    except:
        st.error("Model file 'model.pth' not found. Please train and save your model first.")
    model.eval()
    return model

model = load_model()
categories = ['B', 'O', 'E']

# --- 3. STREAMLIT UI ---
st.title("Character Recognition Web App (B, O, E)")
st.write("Upload an 8x8 image to predict the letter category.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Pre-processing: Convert upload to 8x8 grayscale and flatten to 64 
    from PIL import Image
    img = Image.open(uploaded_file).convert('L').resize((8, 8))
    img_array = np.array(img).astype(np.float32)
    
    # Normalize image to match training data (-1 to 1 range roughly)
    # Note: In a real app, you'd apply the same noise/normalization as training
    processed_input = (img_array / 127.5) - 1.0 
    input_tensor = torch.tensor(processed_input.flatten()).unsqueeze(0)

    # --- 4. PREDICTION  ---
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)

    # --- 5. DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded 8x8 Image", width=150)
    with col2:
        st.success(f"Predicted Letter: **{categories[prediction]}**")
        st.write(f"Confidence: {confidence[0][prediction]*100:.2f}%")

    # Show confidence for all classes
    st.bar_chart({categories[i]: float(confidence[0][i]) for i in range(3)})