# Emotion Detection from Facial Images - Starter Notebook with Streamlit App (Using Folder-Based FER2013)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import streamlit as st
import cv2
import os
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dir = "emotionDetection/train"
test_dir = "emotionDetection/test"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN Model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (single epoch)
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_cnn.pth")
    print("Model saved to emotion_cnn.pth")

# train_one_epoch()

# Emotion label map based on ImageFolder class indices
emotion_map = train_dataset.classes

# Streamlit app
st.title("😃 Real-time Emotion Detection from Facial Images")

model.eval()
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=device))

def predict_emotion(image):
    image = Image.fromarray(image).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return emotion_map[predicted.item()]

option = st.radio("Select input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        pred = predict_emotion(np.array(img.resize((48, 48))))
        st.success(f"Predicted Emotion: {pred}")

elif option == "Use Webcam":
    run = st.button("Start Webcam")
    stop = st.button("Stop")
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    while run and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam error")
            break
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face, (48, 48))
        emotion = predict_emotion(face_resized)
        cv2.putText(frame, emotion, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        frame_placeholder.image(frame, channels="BGR")
    cap.release()
    frame_placeholder.empty()






























# import tensorflow as tf
# import numpy as np
# import streamlit as st
# from pygments import highlight
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Embedding
# import os
# import re
#
#
#
# @st.cache_resource
# def load_mod():
#     model = load_model('reuters_news_classifier.h5')
#     word_index = tf.keras.datasets.reuters.get_word_index()
#     return model, word_index
#
# model, word_index = load_mod()
#
# class_names = {
#     0: 'cocoa', 1: 'grain', 2: 'veg-oil', 3: 'earn', 4: 'acq', 5: 'wheat', 6: 'corn',
#     7: 'money-fx', 8: 'interest', 9: 'ship', 10: 'cotton', 11: 'coffee', 12: 'sugar',
#     13: 'trade', 14: 'reserves', 15: 'veg-oil', 16: 'money-supply', 17: 'tin',
#     18: 'strategic-metal', 19: 'nat-gas', 20: 'cpi', 21: 'housing', 22: 'jobs',
#     23: 'money-fx', 24: 'gold', 25: 'silver', 26: 'lei', 27: 'retail', 28: 'ipi',
#     29: 'carcass', 30: 'livestock', 31: 'orange', 32: 'heat', 33: 'fuel',
#     34: 'gas', 35: 'instal-debt', 36: 'inventories', 37: 'grain', 38: 'meal-feed',
#     39: 'oat', 40: 'rape-oil', 41: 'rubber', 42: 'ship', 43: 'soy-meal',
#     44: 'soy-oil', 45: 'soybean'
# }
#
# def preprocess_text(text, word_index, maxlen=200):
#     tokens = re.findall(r"\b\w+\b", text.lower())
#     sequences = [word_index.get(word, 2) + 3 for word in tokens]
#     paded = pad_sequences([sequences], maxlen=maxlen)
#     return paded, tokens
#
# st.title("News topic classifier")
# user_input = st.text_area("Enter a news article snippet", "The crypto market is in danger.")
#
# if st.button("Classify"):
#     seq, tokens = preprocess_text(user_input, word_index)
#     prediction = model.predict(seq)[0]
#
#     # Top three prediction
#     top3_indices = prediction.argsort()[-3:][::-1]
#     top3_labels = [(class_names.get(i, f"Class {i}"), prediction[i]) for i in top3_indices]
#
#     #Top prediciton
#     top_label, top_confidence = top3_labels[0]
#     st.subheader("🔎 Prediction Result")
#     st.markdown(f"**Top Topic:** `{top_label}` — {top_confidence:.2%} confidence")
#
#     st.markdown("---")
#     st.markdown("**Top three predictions:**")
#     for label, conf in top3_labels:
#         st.write(f"-{label}: {conf:.2%}")
#
#     highlighted_text = []
#     for word in tokens:
#         if word in word_index:
#             highlighted_text.append("**{word}**")
#         else:
#             highlighted_text.append(word)
#
#     st.markdown("---")
#     st.markdown("**Highlighted Input**")
#     st.write(" ".join(highlighted_text))
#