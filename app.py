import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import json

# ==============================
# Load Config + Classes
# ==============================
# Agar Kaggle se `training_results.json` bhi download kiya hai to usse classes lo
try:
    with open("training_results.json", "r") as f:
        results = json.load(f)
    CONFIG = results["config"]
    classes = results["classes"]
except:
    # fallback (agar json nahi hai to manually likho)
    CONFIG = {
        "model_name": "convnext_tiny",
        "img_size": 224,
        "drop_path_rate": 0.2
    }
    classes = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", "Bhadawari", 
    "Brown_Swiss", "Dangi", "Deoni", "Gir", "Guernsey", "Hallikar", "Hariana", 
    "Holstein_Friesian", "Jaffrabadi", "Jersey", "Kangayam", "Kankrej", 
    "Kasargod", "Kenkatha", "Kherigarh", "Khillari", "Krishna_Valley", 
    "Malnad_gidda", "Mehsana", "Murrah", "Nagori", "Nagpuri", "Nili_Ravi", 
    "Nimari", "Ongole", "Pulikulam", "Rathi", "Red_Dane", "Red_Sindhi", 
    "Sahiwal", "Surti", "Tharparkar", "Toda", "Umblachery", "Vechur"
]

num_classes = len(classes)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# Model Load
# ==============================
def load_model(pth_file):
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=CONFIG.get("drop_path_rate", 0.2)
    )
    checkpoint = torch.load(pth_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model("best_model_final.pth")

# ==============================
# Image Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# Prediction Function
# ==============================
def predict(img: Image.Image):
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    return classes[pred_idx.item()], conf.item()

# ==============================
# Streamlit UI
# ==============================
st.title("🐄 Indian Cow Breed Classifier")
st.write("Upload an image of a cow to predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        breed, confidence = predict(img)

    st.success(f"**Predicted Breed:** {breed}")
    st.info(f"Confidence: {confidence:.2f}")
