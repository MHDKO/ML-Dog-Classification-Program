from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.camera import Camera

import os
import base64
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load model and class names
model = None
class_names = []

def load_resources():
    global model, class_names
    try:
        # Enable eager execution
        tf.config.run_functions_eagerly(True)

        # Load model
        model_path = "final_dog_inception_model.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        else:
            print(f"Model file not found at {model_path}")

        # Load labels
        labels_path = "labels.csv"
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            class_names = sorted(df["breed"].unique())
            print("Labels loaded successfully")
        else:
            print(f"Labels file not found at {labels_path}")
    except Exception as e:
        print(f"Error loading resources: {e}")

def predict_dog_breed(img):
    if model is None or not class_names:
        return "Model or labels not loaded"
    try:
        # Preprocess image
        img = img.resize((299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Make prediction with memory management
        with tf.device('/CPU:0'):  # Force CPU usage to prevent GPU memory issues
            try:
                # Run prediction in a try block
                preds = model(img_tensor, training=False)
                preds = preds.numpy()  # Convert to numpy array
                top_idx = np.argmax(preds)
                breed = class_names[top_idx]
                confidence = preds[0][top_idx]

                # Clear memory
                tf.keras.backend.clear_session()
                return f"{breed} (Confidence: {confidence:.2%})"
            except Exception as pred_error:
                tf.keras.backend.clear_session()
                return f"Prediction computation error: {str(pred_error)}"
    except Exception as e:
        tf.keras.backend.clear_session()
        return f"Image processing error: {str(e)}"

def vet_chatbot(user_input, history):
    if not API_KEY:
        return "API Key not found"
    try:
        messages = [{"role": "system", "content": "You are a helpful veterinary assistant."}]
        for user_msg, bot_msg in history:
            messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": bot_msg}
            ])
        messages.append({"role": "user", "content": user_input})

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": "deepseek/deepseek-chat:free", "messages": messages}

        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = None
        self.captured_image = None

        # Main layout
        layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        # Title
        title = Label(
            text="Take Photo",
            font_size=dp(24),
            size_hint_y=0.1,
            color=get_color_from_hex('#2E7D32'),
            bold=True
        )
        layout.add_widget(title)

        # Camera preview
        self.camera_preview = BoxLayout(size_hint_y=0.7)
        layout.add_widget(self.camera_preview)

        # Buttons
        button_row = BoxLayout(
            size_hint_y=0.2,
            spacing=dp(10)
        )

        predict_btn = Button(
            text="Predict Breed",
            background_color=get_color_from_hex('#2196F3'),
            background_normal=''
        )
        cancel_btn = Button(
            text="Cancel",
            background_color=get_color_from_hex('#F44336'),
            background_normal=''
        )

        predict_btn.bind(on_press=self.predict_from_camera)
        cancel_btn.bind(on_press=self.cancel_camera)

        button_row.add_widget(predict_btn)
        button_row.add_widget(cancel_btn)
        layout.add_widget(button_row)

        self.add_widget(layout)

    def on_enter(self):
        # Start camera when screen is entered
        self.start_camera()

    def on_leave(self):
        # Stop camera when leaving screen
        self.stop_camera()

    def start_camera(self):
        try:
            self.camera = Camera(play=True, resolution=(640, 480))
            self.camera_preview.clear_widgets()
            self.camera_preview.add_widget(self.camera)
        except Exception as e:
            print(f"Error starting camera: {e}")

    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_preview.clear_widgets()
            self.camera = None

    def predict_from_camera(self, instance):
        if self.camera:
            try:
                # Capture the current frame
                texture = self.camera.texture
                if texture:
                    # Convert texture to PIL Image
                    size = texture.size
                    pixels = texture.pixels
                    image_data = bytes(pixels)
                    image = PILImage.frombytes('RGBA', size, image_data)
                    image = image.convert('RGB')

                    # Save to temporary file
                    temp_path = os.path.join(App.get_running_app().user_data_dir, 'temp_camera.png')
                    image.save(temp_path)

                    # Store the captured image
                    self.captured_image = image

                    # Get prediction
                    result = predict_dog_breed(image)

                    # Return to classifier screen with result
                    classifier_screen = self.manager.get_screen('classifier')
                    classifier_screen.handle_captured_image(image, temp_path)
                    classifier_screen.result_label.text = result
                    self.manager.current = 'classifier'
            except Exception as e:
                print(f"Error capturing and predicting: {e}")

    def cancel_camera(self, instance):
        self.manager.current = 'classifier'

class DogClassifierScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_image = None
        self.img_copy = None

        # Main layout
        layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        # Title
        title = Label(
            text="Dog Breed Classifier",
            font_size=dp(24),
            size_hint_y=0.1,
            color=get_color_from_hex('#2E7D32'),
            bold=True
        )
        layout.add_widget(title)

        # Image preview
        self.image_preview = KivyImage(
            size_hint_y=0.4,
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(self.image_preview)

        # Buttons
        button_row = BoxLayout(
            size_hint_y=0.15,
            spacing=dp(10)
        )

        upload_btn = Button(
            text="Upload Image",
            background_color=get_color_from_hex('#4CAF50'),
            background_normal=''
        )
        camera_btn = Button(
            text="Take Photo",
            background_color=get_color_from_hex('#9C27B0'),
            background_normal=''
        )
        paste_btn = Button(
            text="Paste Image",
            background_color=get_color_from_hex('#FF9800'),
            background_normal=''
        )
        self.predict_btn = Button(
            text="Predict Breed",
            background_color=get_color_from_hex('#2196F3'),
            background_normal='',
            disabled=True
        )

        upload_btn.bind(on_press=self.show_file_chooser)
        camera_btn.bind(on_press=self.show_camera)
        paste_btn.bind(on_press=self.paste_image)
        self.predict_btn.bind(on_press=self.predict)

        button_row.add_widget(upload_btn)
        button_row.add_widget(camera_btn)
        button_row.add_widget(paste_btn)
        button_row.add_widget(self.predict_btn)
        layout.add_widget(button_row)

        # Result area
        result_container = BoxLayout(
            orientation="vertical",
            size_hint_y=0.35,
            padding=dp(10)
        )

        result_title = Label(
            text="Prediction Result:",
            font_size=dp(18),
            size_hint_y=0.2,
            color=get_color_from_hex('#000000'),
            bold=True
        )

        self.result_label = Label(
            text="",
            font_size=dp(16),
            size_hint_y=0.8,
            color=get_color_from_hex('#000000')
        )

        result_container.add_widget(result_title)
        result_container.add_widget(self.result_label)
        layout.add_widget(result_container)

        self.add_widget(layout)
        self.file_chooser_popup = None

    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserIconView(
            path=os.path.expanduser('~'),
            filters=['*.png', '*.jpg', '*.jpeg']
        )

        buttons = BoxLayout(size_hint_y=0.1)
        select_btn = Button(text='Select')
        cancel_btn = Button(text='Cancel')

        select_btn.bind(on_press=lambda x: self.load_image(file_chooser.selection))
        cancel_btn.bind(on_press=self.dismiss_file_chooser)

        buttons.add_widget(select_btn)
        buttons.add_widget(cancel_btn)

        content.add_widget(file_chooser)
        content.add_widget(buttons)

        self.file_chooser_popup = Popup(
            title='Select Image',
            content=content,
            size_hint=(0.9, 0.9)
        )
        self.file_chooser_popup.open()

    def load_image(self, selection):
        if selection:
            try:
                path = selection[0]
                self.selected_image = PILImage.open(path).convert('RGB')
                self.image_preview.source = path
                self.predict_btn.disabled = False
                self.result_label.text = ""
            except Exception as e:
                self.result_label.text = f"Error loading image: {str(e)}"
        self.dismiss_file_chooser()

    def paste_image(self, instance):
        try:
            content = Clipboard.paste()
            if not content:
                self.result_label.text = "Clipboard is empty"
                return

            try:
                # Try to decode base64 image
                image_data = base64.b64decode(content)
                img = PILImage.open(io.BytesIO(image_data)).convert('RGB')
            except:
                # Try to treat as file path
                if os.path.exists(content) and content.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = PILImage.open(content).convert('RGB')
                else:
                    self.result_label.text = "Invalid clipboard content"
                    return

            # Save temporary file
            temp_path = os.path.join(App.get_running_app().user_data_dir, 'temp_paste.png')
            img.save(temp_path)

            self.selected_image = img
            self.image_preview.source = temp_path
            self.predict_btn.disabled = False
            self.result_label.text = ""

        except Exception as e:
            self.result_label.text = f"Error pasting image: {str(e)}"

    def predict(self, instance):
        if not self.selected_image:
            self.result_label.text = "No image selected"
            return

        self.result_label.text = "Predicting..."
        self.predict_btn.disabled = True  # Disable button during prediction

        try:
            # Create a copy of the image
            self.img_copy = self.selected_image.copy()

            def run_prediction():
                try:
                    # Run prediction in a separate thread
                    result = predict_dog_breed(self.img_copy)
                    Clock.schedule_once(lambda dt: self._update_result(result))
                except Exception as pred_error:
                    error_msg = f"Thread error: {str(pred_error)}"
                    Clock.schedule_once(lambda dt: self._update_result(error_msg))
                finally:
                    if hasattr(self, 'img_copy') and self.img_copy is not None:
                        del self.img_copy
                        self.img_copy = None
                    import gc
                    gc.collect()

            # Start prediction in a separate thread
            import threading
            thread = threading.Thread(target=run_prediction, daemon=True)
            thread.start()

        except Exception as e:
            self.result_label.text = f"Error starting prediction: {str(e)}"
            self.predict_btn.disabled = False

    def _update_result(self, result):
        try:
            self.result_label.text = result
        except Exception as e:
            self.result_label.text = f"Error updating result: {str(e)}"
        finally:
            self.predict_btn.disabled = False  # Re-enable button after prediction

    def dismiss_file_chooser(self, *args):
        if self.file_chooser_popup:
            self.file_chooser_popup.dismiss()
            self.file_chooser_popup = None

    def show_camera(self, instance):
        self.manager.current = 'camera'

    def handle_captured_image(self, image, path):
        self.selected_image = image
        self.image_preview.source = path
        self.predict_btn.disabled = False
        self.result_label.text = ""

class VetAssistantScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []

        # Main layout
        layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        # Title
        title = Label(
            text="Veterinary Assistant",
            font_size=dp(24),
            size_hint_y=0.1,
            color=get_color_from_hex('#2E7D32'),
            bold=True
        )
        layout.add_widget(title)

        # Chat area
        chat_container = BoxLayout(orientation="vertical", size_hint_y=0.8)

        # Scrollable chat log
        self.scroll = ScrollView(size_hint_y=0.9)
        self.chat_log = Label(
            text="",
            size_hint_y=None,
            text_size=(Window.width - dp(40), None),
            markup=True,
            padding=dp(10),
            color=get_color_from_hex('#000000')  # Set text color to black
        )
        self.chat_log.bind(texture_size=self.update_chat_height)
        self.scroll.add_widget(self.chat_log)
        chat_container.add_widget(self.scroll)

        # Input area
        input_container = BoxLayout(
            size_hint_y=0.1,
            spacing=dp(10)
        )

        self.input = TextInput(
            hint_text="Ask your question...",
            multiline=False,
            padding=dp(10),
            size_hint_x=0.8,
            foreground_color=get_color_from_hex('#000000')  # Set input text color to black
        )

        send_btn = Button(
            text="Send",
            size_hint_x=0.2,
            background_color=get_color_from_hex('#4CAF50'),
            background_normal=''
        )
        send_btn.bind(on_press=self.ask_bot)

        input_container.add_widget(self.input)
        input_container.add_widget(send_btn)
        chat_container.add_widget(input_container)

        layout.add_widget(chat_container)
        self.add_widget(layout)

    def update_chat_height(self, instance, value):
        instance.height = value[1]
        self.scroll.scroll_y = 0

    def ask_bot(self, instance):
        question = self.input.text.strip()
        if question:
            # Add user message
            self.chat_log.text += f"[b]You:[/b] {question}\n"
            self.input.text = ""

            # Get bot response
            Clock.schedule_once(lambda dt: self._get_bot_response(question))

    def _get_bot_response(self, question):
        try:
            response = vet_chatbot(question, self.history)
            self.chat_log.text += f"[b]Vet:[/b] {response}\n\n"
            self.history.append((question, response))
        except Exception as e:
            self.chat_log.text += f"[b]Error:[/b] {str(e)}\n\n"

class MainApp(App):
    def build(self):
        # Load resources
        load_resources()

        # Set window color
        Window.clearcolor = get_color_from_hex('#F5F5F5')

        # Create screen manager
        sm = ScreenManager()

        # Add screens
        sm.add_widget(DogClassifierScreen(name="classifier"))
        sm.add_widget(VetAssistantScreen(name="vet"))
        sm.add_widget(CameraScreen(name="camera"))

        # Main layout
        root = BoxLayout(orientation="vertical")

        # Tab bar
        tab_bar = BoxLayout(
            size_hint_y=0.1,
            spacing=dp(5),
            padding=dp(5)
        )

        # Create tab buttons
        btn1 = Button(
            text="Dog Classifier",
            background_color=get_color_from_hex('#4CAF50'),
            background_normal=''
        )
        btn2 = Button(
            text="Vet Assistant",
            background_color=get_color_from_hex('#2196F3'),
            background_normal=''
        )

        # Bind tab switching
        btn1.bind(on_press=lambda x: self.switch_screen(sm, "classifier", btn1, btn2))
        btn2.bind(on_press=lambda x: self.switch_screen(sm, "vet", btn2, btn1))

        tab_bar.add_widget(btn1)
        tab_bar.add_widget(btn2)

        root.add_widget(tab_bar)
        root.add_widget(sm)

        return root

    def switch_screen(self, screen_manager, screen_name, active_btn, inactive_btn):
        screen_manager.current = screen_name
        active_btn.background_color = get_color_from_hex('#2E7D32')
        inactive_btn.background_color = get_color_from_hex('#1976D2')

if __name__ == "__main__":
    MainApp().run()
