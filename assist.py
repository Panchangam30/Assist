import os
import time
import base64
import openai
import pygame
import re
import pyautogui
import cv2  # OpenCV for image processing
import pytesseract
from PIL import Image
from gtts import gTTS
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import speech_recognition as sr

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

class Jarvis:
    def __init__(self):
        self.screen_text = ""

    def interpret_command(self, command):
        """Use GPT-4 to interpret the user's input command and return a structured action."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an assistant that understands natural language commands. "
                            "Your task is to classify the following user command into one of these actions: "
                            "'send_email', 'create_event', 'exit', 'look_at_screen', or 'answer_question'. "
                            "Respond with only one of these exact actions and nothing else. "
                            f"Here is the user's command: {command}"
                        )
                    }
                ],
                max_tokens=5,
                temperature=0.0,
            )
            action = response.choices[0].message['content'].strip().lower()
            print(f"Action interpreted: {action}")
            return action

        except Exception as e:
            print(f"An error occurred while interpreting the command: {e}")
            return "unknown_action"

    def handle_command(self, command):
        """Handle the interpreted command and execute the appropriate action."""
        print(f"Received command: {command}")
        action = self.interpret_command(command)

        if action == 'send_email':
            self.send_email()
        elif action == 'create_event':
            self.create_calendar_event()
        elif action == 'exit':
            self.speak("Goodbye!")
            exit(0)
        elif action == 'look_at_screen':
            self.look_at_screen()
        elif action == 'answer_question':
            self.ask_question_about_screen(command)
        else:
            self.speak("I don't understand the command. Please try again.")

    def send_email(self):
        """Send an email."""
        print("Preparing to send an email...")
        recipient = input("Enter recipient email: ")
        subject = input("Enter email subject: ")
        body = input("Enter email body: ")

        message = MIMEText(body)
        message['to'] = recipient
        message['from'] = "your-email@gmail.com"  # Replace with your Gmail
        message['subject'] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        try:
            print(f"Email sent to {recipient}.")
        except Exception as e:
            self.speak("Failed to send the email.")
            print(f"An error occurred: {e}")

    def create_calendar_event(self):
        """Create a new event in the user's Google Calendar."""
        self.speak("Let's create a calendar event.")
        title = input("What is the title of the event? ")
        date_str = input("When is the event (YYYY-MM-DD)? ")
        time_str = input("At what time (HH:MM)? ")

        start_time = f"{date_str}T{time_str}:00"
        end_time = (datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S") + timedelta(hours=1)).isoformat()

        event = {
            'summary': title,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }

        try:
            print(f"Event '{title}' has been created.")
        except Exception as e:
            self.speak("Failed to create the event.")
            print(f"An error occurred: {e}")

    def preprocess_image(self, image_path):
        """Preprocess the image for better text recognition."""
        image = cv2.imread(image_path)

        # Convert to grayscale for better contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a black and white image
        _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Optionally resize the image for better OCR accuracy
        resized_image = cv2.resize(binary_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # Apply some denoising
        denoised = cv2.fastNlMeansDenoising(resized_image, h=30)

        return denoised

    def look_at_screen(self):
        """Capture the screen and analyze the image using OpenCV and pytesseract."""
        self.speak("Looking at your screen...")
        screenshot = pyautogui.screenshot()
        screenshot_path = "screenshot.png"
        screenshot.save(screenshot_path)

        # Preprocess the image for better OCR
        preprocessed_image = self.preprocess_image(screenshot_path)

        # Extract text using pytesseract from the preprocessed image
        self.screen_text = pytesseract.image_to_string(preprocessed_image)
        
        # Clean up the extracted text to remove artifacts and improve quality
        self.screen_text = self.clean_extracted_text(self.screen_text)
        print(f"Cleaned Extracted Text: {self.screen_text}")

        # Proceed to answering questions based on the cleaned text
        self.speak("What would you like me to answer?")
        question = self.listen()
        self.ask_question_about_screen(question)

    def clean_extracted_text(self, text):
        """Clean the extracted text by removing unnecessary symbols and artifacts."""
        # Remove any special characters or misrecognized text artifacts
        clean_text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)  # Remove non-alphanumeric characters
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
        return clean_text.strip()

    def ask_question_about_screen(self, question):
        """Process the user's question about the screen content."""
        print(f"Processing question about screen: {question}")

        try:
            # Use both the extracted text and GPT's general knowledge to answer
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that can answer questions based on both the extracted text from images and your own knowledge. "
                        "Use the extracted text to help guide your answer, but feel free to use your general knowledge to provide a complete, well-rounded response."
                    )},
                    {"role": "user", "content": f"Based on the following extracted text: '{self.screen_text}', and any other relevant knowledge, answer the question: '{question}'"},
                ],
                max_tokens=200,
            )
            answer = response.choices[0].message['content'].strip()
            self.speak(f"The answer is: {answer}")
        except Exception as e:
            self.speak("Failed to answer the question.")
            print(f"An error occurred while answering the question: {e}")

    def listen(self):
        """Listen for the user's voice input."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Heard: {command}")
                return command
            except sr.UnknownValueError:
                self.speak("I didn't catch that. Please try again.")
                return ""
            except sr.RequestError:
                self.speak("There was an error with the speech recognition service.")
                return ""

    def speak(self, text):
        """Convert text to speech and play it."""
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        os.remove("response.mp3")

def main():
    jarvis = Jarvis()

    while True:
        command = jarvis.listen()
        if command:
            jarvis.handle_command(command)

if __name__ == "__main__":
    main()