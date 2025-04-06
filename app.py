from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import webbrowser
import pyttsx3
import google.generativeai as genai
from gtts import gTTS
import pygame
import os
from dotenv import load_dotenv
import markdown  # Import the markdown library

load_dotenv()

app = Flask(__name__)
app.config['CONVERSATION_HISTORY'] = []  # In-memory storage for conversation
app.config['AUDIO_PLAYING'] = False  # Track if audio is currently playing

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    model = None

recognizer = sr.Recognizer()
engine_tts = pyttsx3.init()

def speak_old(text):
    engine_tts.say(text)
    engine_tts.runAndWait()

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save('temp.mp3')
        pygame.mixer.init()
        pygame.mixer.music.load('temp.mp3')
        pygame.mixer.music.play()
        app.config['AUDIO_PLAYING'] = True  # Set flag to True
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        os.remove("temp.mp3")
        app.config['AUDIO_PLAYING'] = False # Reset flag
    except Exception as e:
        print(f"Error during speech: {e}")
        app.config['AUDIO_PLAYING'] = False # Ensure flag is reset on error

def stop_audio():
    if app.config['AUDIO_PLAYING']:
        pygame.mixer.music.stop()
        app.config['AUDIO_PLAYING'] = False
        return "Audio stopped."
    else:
        return "No audio is currently playing."

def aiProcess(command, history=None):
    if model:
        try:
            prompt = command
            if history:
                context = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history])
                prompt = f"{context}\nUser: {command}\nAI:"
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return "Error communicating with AI."
    else:
        return "AI model not initialized."

def processCommand(c):
    command_lower = c.lower()
    if "stop" in command_lower:
        return stop_audio()
    else:
        history = app.config['CONVERSATION_HISTORY']
        output = aiProcess(c, history)
        app.config['CONVERSATION_HISTORY'].append({'user': c, 'ai': output})
        # Keep only the last few turns to prevent overly long prompts
        app.config['CONVERSATION_HISTORY'] = app.config['CONVERSATION_HISTORY'][-5:]
        return output

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        command = request.form.get("command")
        if command:
            ai_response = processCommand(command)
            response = markdown.markdown(ai_response)
    return render_template("index.html", response=response)

@app.route('/listen', methods=['POST'])
def listen_endpoint():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        r.adjust_for_ambient_noise(source, duration=0.2)

        try:
            audio = r.listen(source, timeout=5)
            command = r.recognize_google(audio)
            print(f"Command heard: {command}")
            ai_response = processCommand(command)
            response = markdown.markdown(ai_response)
            speak(ai_response)  # Speak the original response
            return jsonify({'response': response})

        except sr.WaitTimeoutError:
            return jsonify({'response': 'Timeout - No audio detected'})
        except sr.UnknownValueError:
            return jsonify({'response': 'Could not understand audio'})
        except sr.RequestError as e:
            error_message = f"Could not request results from speech recognition service; {e}"
            print(error_message)
            return jsonify({'response': error_message})
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            return jsonify({'response': error_message})

@app.route('/stop_audio', methods=['POST'])
def stop_audio_endpoint():
    message = stop_audio()
    return jsonify({'response': message})

if __name__ == "__main__":
    print("Flask app started. Make sure you have pygame installed (`pip install pygame`) and the `markdown` library (`pip install markdown`).")
    app.run(debug=True)