import gradio as gr
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY =''  # .env must contain OPENROUTER_API_KEY=sk-...

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("RuntimeError during GPU memory growth setup:", e)

# Load model and breed labels
model = tf.keras.models.load_model('final_dog_inception_model.keras')
df = pd.read_csv('/labels.csv')
class_names = sorted(df['breed'].unique())


def predict_dog_breed(img):
    try:
        img = img.resize((299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        preds = model.predict(img_array)
        top_idx = np.argmax(preds)
        breed = class_names[top_idx]
        confidence = preds[0][top_idx]
        return f"{breed} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"Prediction error: {e}"

def vet_chatbot(user_input, history=[]):
    try:
        API_URL = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        # Construct message history
        messages = [{"role": "system", "content": "You are a helpful and accurate veterinary assistant. Keep your responses clear and concise."}]
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": user_input})

        # Payload
        data = {
            "model": "deepseek/deepseek-chat:free",
            "messages": messages
        }

        # API Call
        response = requests.post(API_URL, json=data, headers=headers)

        if response.status_code == 200:
            api_response = response.json()
            answer = api_response['choices'][0]['message']['content'].strip()
        else:
            answer = f"Failed to fetch data from API. Status Code: {response.status_code}"

        history.append((user_input, answer))
        return history, history

    except Exception as e:
        error_msg = f"Error: {e}"
        history.append((user_input, error_msg))
        return history, history


dog_interface = gr.Interface(
    fn=predict_dog_breed,
    inputs=gr.Image(type="pil", label="Upload Dog Image"),
    outputs=gr.Text(label="Predicted Breed"),
    title="Dog Breed Classifier",
    description="Upload an image of a dog to identify its breed."
)

chatbot = gr.Chatbot(label="Vet Assistant")
state = gr.State([])

def vet_chat(user_input, history):
    return vet_chatbot(user_input, history)

vet_interface = gr.Interface(
    fn=vet_chat,
    inputs=[gr.Textbox(placeholder="Ask a question about your dog...", label="Your Question"), state],
    outputs=[chatbot, state],
    title="Vet Assistant Chatbot",
    description="Ask about dog care, health, or training."
)

app = gr.TabbedInterface([dog_interface, vet_interface], tab_names=["Dog Classifier", "Vet Assistant"])
app.launch()
