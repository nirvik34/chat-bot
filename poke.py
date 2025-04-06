from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import requests
from gtts import gTTS
import pygame
import os
import markdown
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.config['CONVERSATION_HISTORY'] = []
app.config['AUDIO_PLAYING'] = False

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "mixtral-8x7b-32768"  # You can change this to another supported model

# Text-to-Speech using gTTS
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        app.config['AUDIO_PLAYING'] = True

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
        os.remove("temp.mp3")
        app.config['AUDIO_PLAYING'] = False
    except Exception as e:
        print(f"Speech error: {e}")
        app.config['AUDIO_PLAYING'] = False

# Stop audio playback
def stop_audio():
    if app.config['AUDIO_PLAYING']:
        pygame.mixer.music.stop()
        app.config['AUDIO_PLAYING'] = False
        return "Audio stopped."
    return "No audio is currently playing."

# Communicate with Groq API
def aiProcess(command, history=None):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is missing in environment variables."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["ai"]})

    messages.append({"role": "user", "content": command})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq API Error: {e}")
        return "Failed to get response from Groq AI."

# Logic to process the command
def processCommand(command):
    command_lower = command.lower()
    if "stop" in command_lower:
        return stop_audio()
    else:
        history = app.config['CONVERSATION_HISTORY']
        response = aiProcess(command, history)
        app.config['CONVERSATION_HISTORY'].append({'user': command, 'ai': response})
        app.config['CONVERSATION_HISTORY'] = app.config['CONVERSATION_HISTORY'][-5:]
        return response

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        command = request.form.get("command")
        if command:
            ai_response = processCommand(command)
            response = markdown.markdown(ai_response)
    return render_template("index.html", response=response)

# Voice input route
@app.route("/listen", methods=["POST"])
def listen_endpoint():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.2)

        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            print(f"🗣️ Heard: {command}")
            ai_response = processCommand(command)
            response = markdown.markdown(ai_response)
            speak(ai_response)
            return jsonify({"response": response})
        except sr.WaitTimeoutError:
            return jsonify({"response": "Timeout - No audio detected"})
        except sr.UnknownValueError:
            return jsonify({"response": "Could not understand audio"})
        except sr.RequestError as e:
            return jsonify({"response": f"Speech recognition service error: {e}"})
        except Exception as e:
            return jsonify({"response": f"Unexpected error: {e}"})

# Stop audio route
@app.route("/stop_audio", methods=["POST"])
def stop_audio_endpoint():
    message = stop_audio()
    return jsonify({"response": message})

# Run the app
if __name__ == "__main__":
    print("✅ JurisBot Flask server running with Groq API.")
    app.run(debug=True)
