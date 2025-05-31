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
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU configuration error:", e)

# Load model and breed labels
try:
    model = tf.keras.models.load_model('final_dog_inception_model.keras')
    df = pd.read_csv('/labels.csv')
    class_names = sorted(df['breed'].unique())
except Exception as e:
    print(f"Error loading model or labels: {e}")
    raise

def predict_dog_breed(img):
    """Predict dog breed from image with error handling and confidence score"""
    if img is None:
        return "Please upload an image"

    try:
        # Preprocess image
        img = img.resize((299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        top_idx = np.argmax(preds)
        confidence = float(preds[0][top_idx])
        breed = class_names[top_idx]

        # Get top 3 predictions
        top_3_idx = np.argsort(preds[0])[-3:][::-1]
        results = []
        for idx in top_3_idx:
            breed_name = class_names[idx]
            conf = float(preds[0][idx])
            results.append(f"{breed_name.replace('_', ' ').title()}: {conf*100:.1f}%")

        return "\n".join(results)

    except Exception as e:
        return f"Error processing image: {str(e)}"

def vet_chatbot(message, history):
    """Enhanced vet chatbot with better error handling"""
    if not message:
        return history

    try:
        # Prepare chat API request
        API_URL = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        # Create context from history
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable and friendly veterinary assistant. Provide clear, accurate advice about dog care, health, and training. Keep responses concise and practical."
            }
        ]

        # Add history to context
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Make API request
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "model": "deepseek/deepseek-chat:free",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
        )

        response.raise_for_status()  # Raise exception for bad status codes
        answer = response.json()['choices'][0]['message']['content'].strip()

        history.append((message, answer))
        return history

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        history.append((message, error_msg))
        return history
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return history

# Create mobile-friendly interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🐕 Dog Breed Classifier & Vet Assistant")

    with gr.Tab("Dog Classifier"):
        with gr.Column():
            gr.Markdown("## Upload a dog photo to identify its breed")
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "webcam"]
            )
            with gr.Row():
                classify_btn = gr.Button("Classify", variant="primary")
                clear_btn = gr.Button("Clear")

            output = gr.Textbox(
                label="Predictions",
                placeholder="Upload an image and click Classify to see predictions...",
                lines=4
            )

            classify_btn.click(
                fn=predict_dog_breed,
                inputs=image_input,
                outputs=output
            )
            clear_btn.click(
                fn=lambda: [None, ""],
                inputs=None,
                outputs=[image_input, output]
            )

    with gr.Tab("Vet Assistant"):
        gr.Markdown("## Chat with our AI Vet Assistant")
        chatbot = gr.Chatbot(
            label="Chat History",
            height=400,
            type="messages"  # Updated to use the new message format
        )
        msg = gr.Textbox(
            label="Type your question",
            placeholder="Ask about dog care, health, or training...",
            lines=2
        )
        clear = gr.Button("Clear Chat")

        msg.submit(vet_chatbot, [msg, chatbot], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app with mobile-friendly configurations
app.launch(
    share=True,
    show_error=True,
    height=800,
    server_name="0.0.0.0"  # Makes it accessible from other devices
)

