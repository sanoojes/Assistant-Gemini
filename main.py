import os
import pickle
import cv2
import face_recognition
import pyttsx3
import google.generativeai as genai
import speech_recognition as sr
from dotenv import load_dotenv  

load_dotenv()

dev_name = os.getenv("DEV_NAME", "Sachu")  
bot_name = os.getenv("BOT_NAME", "Cloudy") 
genai_api_key = os.getenv("GENAI_API_KEY")


if genai_api_key is None:
    raise ValueError("GENAI_API_KEY environment variable not set. Please create a .env file and add your API key.")
genai.configure(api_key=genai_api_key)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

initial_instructions = f"""You are {bot_name}, your voice-based chat assistant, developed by {dev_name}. Respond exclusively through voice interactions, providing concise and relevant information. Avoid generating technical details, code. Focus on clear, direct answers suitable for voice-based communication. If clarification is needed, ask to ensure accuracy. Continuously update knowledge to provide current and accurate responses. Requests should be made in a way that aligns with voice-based answers.Don't use emojis, special charecters, and do not generate code samples.DONT USE MARKDOWN OR ANY SPECIFIC MARKDOWN LANGUAGE."""

model = genai.GenerativeModel(
    # model_name="gemini-1.0-pro", //  does not have dev instructions 
    # model_name="gemini-1.5-pro",
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=initial_instructions
)

chat_session = model.start_chat(history=[])

r = sr.Recognizer()

engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (1 is max)

def listen():
    while True:  
        with sr.Microphone() as source:
            print("Listening...")
            
            r.adjust_for_ambient_noise(source)  
            
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=5) 
                text = r.recognize_google(audio)
                if text:  
                    print(f"You said: {text}")
                    return text  
            except sr.UnknownValueError:
                print("Couldnt understand Listening again.")
            except sr.RequestError as e:
                print(f"Sorry, there was a problem with the service: {e}")
                return ""  
            except Exception as e:
                print(f"An error occurred: {e}")
                return "" 


def speak(text):
    print("AI: ", text)
    engine.say(text)
    engine.runAndWait()

def save_face_encoding(name, face_encoding):
    with open(f"known_faces/{name}.pkl", "wb") as f:
        pickle.dump(face_encoding, f)

def recognize_and_learn_face():
    video_capture = cv2.VideoCapture(0)  

    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir("known_faces"):
        if filename.endswith(".pkl"):
            with open(f"known_faces/{filename}", "rb") as f:
                encoding = pickle.load(f)
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split(".")[0])  
    
    while True:
        ret, frame = video_capture.read()  
        if not ret:
            print("Failed to grab frame.")
            break
        
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Process each face found
        for (top,left,right,bottom), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            if name == "Unknown":
                speak("I don't recognize you. Please provide your name.")
                
                user_name = input("Please enter your name: ") 
                save_face_encoding(user_name, face_encoding)
                speak(f"Nice to meet you, {user_name}!")
                
                known_face_encodings.append(face_encoding)
                known_face_names.append(user_name)
                cv2.destroyAllWindows()
                return user_name  
            else:
                print(f"User detected, Name: {name}!")
                cv2.destroyAllWindows()
                return name  
    
    video_capture.release()
    return None  


user_name = "User" # initial username  
def change_user_name():

    user_name = recognize_and_learn_face()

    chat_session.history.append({"role": "user", "parts": [f"My name is {user_name}"]})

    chat_session.history.append({"role": "model", "parts": [f"User name is changed to {user_name}"]})
    if user_name:
        print(f"User changed to {user_name}!")
        speak(f"Hey {user_name}. Nice to meet you.")        

    elif user_name == "User":
        speak("I can't recognise you i will call you User from now on")
    else:
        speak("No user detected.")
        
def main():
    try:
        speak("Trying to recognise user face.")
        change_user_name()

        while True:
            user_input = listen()
            if user_input.lower() in ["exit", "quit", "bye"]:
                speak("Goodbye!")
                break
            if user_input.lower() in ["change user","user change"]:
                speak("Trying to change user.")
                change_user_name()
                continue

            if user_input:
                chat_session.history.append({"role": "user", "parts": [user_input]}) 
                response = chat_session.send_message(user_name+":"+user_input)
                speak(response.text)
                chat_session.history.append({"role": "model", "parts": [response.text]})

    except KeyboardInterrupt:
        print("Closing Program.")

    
if __name__ == "__main__":
    main()
