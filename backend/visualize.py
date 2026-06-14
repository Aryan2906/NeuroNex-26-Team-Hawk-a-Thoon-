from model import CNNtoSNNConverter
from train import DeepSNNClassifier
import torch

import cv2
import numpy as np
from PIL import Image
from snntorch import utils
import sys
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
sys.path.append(os.path.abspath("../backend"))


client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_patient_report(heatmap_path, snn_diagnosis, confidence):
    try:
        base64_image = encode_image(heatmap_path)
    except FileNotFoundError:
        return

    system_prompt = (
        "You are an elite neurological AI assistant specializing in Explainable AI (XAI) for MRI scans. "
        "The user will provide a Grad-CAM heatmap generated over an axial T1-weighted brain MRI, along with an initial SNN prediction. "
        "Red/Yellow areas indicate high neural activation (areas driving the diagnosis, typically atrophy or enlarged ventricles). "
        "Blue areas indicate normal tissue. "
        "Your job is to write a highly professional, clinical summary report explaining WHAT the heatmap is highlighting and WHY it supports the diagnosis. "
        "Keep it structured, objective, and empathetic. Do not make a definitive medical diagnosis; state that this is an AI screening tool."
    )

    human_prompt = f"The Spiking Neural Network predicted '{snn_diagnosis}' with {confidence:.2f}% confidence. Please analyze the attached Grad-CAM heatmap and generate the patient report."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": human_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.3,
        )

        report = response.choices[0].message.content

        st.subheader("AI Generated Patient Report")
        st.write(report)

        with open("final_patient_report.txt", "w") as f:
            f.write(report)

    except Exception as e:
        print(f"API Communication Failed: {e}")


def generate_heatmap_and_report(
    image_path, retina_weights="snn_retina.pth", brain_weights="snn_brain.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        image_pil = Image.open(image_path).convert("L")
        img_array = np.array(image_pil)
        img_resized = cv2.resize(img_array, (128, 128), interpolation=cv2.INTER_AREA)
        original_bg = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_resized)
        img_norm = img_clahe.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    retina = CNNtoSNNConverter().to(device)
    brain = DeepSNNClassifier().to(device)
    retina.load_state_dict(
        torch.load(retina_weights, map_location=device, weights_only=True)
    )
    brain.load_state_dict(
        torch.load(brain_weights, map_location=device, weights_only=True)
    )

    retina.eval()
    brain.eval()

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target_layer = retina.cnn_extractor[5]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    classes = [
        "Mild_Dementia",
        "Moderate_Dementia",
        "Non_Demented",
        "Very_Mild_Dementia",
    ]
    spk_rec = []

    utils.reset(retina)
    utils.reset(brain)

    for step in range(15):
        spikes_2048, _ = retina(img_tensor)
        class_spikes, _ = brain(spikes_2048)
        spk_rec.append(class_spikes)

    spk_rec = torch.stack(spk_rec)
    total_spikes = spk_rec.sum(dim=0).squeeze(0)

    predicted_idx = total_spikes.argmax()
    confidence = (total_spikes[predicted_idx] / total_spikes.sum()) * 100
    diagnosis = classes[predicted_idx]

    st.success("Analysis Complete")
    st.write(f"### Diagnosis: {diagnosis}")
    st.write(f"### Confidence: {confidence:.2f}%")

    brain.zero_grad()
    retina.zero_grad()
    target_score = total_spikes[predicted_idx]
    target_score.backward()

    if activations is not None and gradients is not None:
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_bg, 0.6, heatmap_colored, 0.4, 0)

        output_filename = "brain_atrophy_analysis.jpg"
        cv2.imwrite(output_filename, superimposed_img)
        st.image(
            output_filename,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

        generate_patient_report(output_filename, diagnosis, confidence)
    else:
        print(" Failed to extract gradients. Check network architecture hooks.")


st.title("🧠 NeuroVision AI")

uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(
        uploaded_file,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("Analyze MRI"):

        with open("uploaded_scan.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        generate_heatmap_and_report("uploaded_scan.jpg")
