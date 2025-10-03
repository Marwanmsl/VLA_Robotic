import os
import cv2
from google import genai
import tkinter as tk
from google.genai import types
from PIL import Image, ImageTk
import json
import time
import threading

# Set your API key
os.environ["GOOGLE_API_KEY"] = "APIkey"

# Initialize client
client = genai.Client()
MODEL_ID = "gemini-robotics-er-1.5-preview"
PROMPT = """You are an object and scene understanding system. 
Return only a JSON response in the format:
{
  "objects": ["object1", "object2", ...],
  "scenario": "Short description of what is happening"
}"""

# Tkinter setup
root = tk.Tk()
root.title("Object & Scene Detection")

# Video frame
video_label = tk.Label(root)
video_label.pack()

# Text area for objects and scenario
text_box = tk.Text(root, height=15, width=199)
text_box.pack(pady=10)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Shared state
last_objects = []
last_scenario = ""
last_api_call = 0
api_interval = 4.5 # seconds


def call_api(img_pil):
    global last_objects, last_scenario, last_api_call
    try:
        image_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img_pil, PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        if image_response.text:
            try:
                data = json.loads(image_response.text)
                last_objects = data.get("objects", [])
                last_scenario = data.get("scenario", "")
            except json.JSONDecodeError:
                last_scenario = "ParseError"
    except Exception as e:
        last_scenario = f"Error: {str(e)}"

    last_api_call = time.time()


def update_frame():
    global last_api_call

    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frame)
        return

    # Resize frame
    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (800, int(800 * height / width)))

    # Convert to PIL for API use
    img_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # Call API asynchronously every N seconds
    if time.time() - last_api_call > api_interval:
        threading.Thread(target=call_api, args=(img_pil,), daemon=True).start()
        last_api_call = time.time()

    # Update text box
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, "Objects:\n")
    for obj in last_objects:
        text_box.insert(tk.END, f"- {obj}\n")
    text_box.insert(tk.END, f"\nScenario:\n{last_scenario}")

    # Show video in Tkinter
    tk_img = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = tk_img
    video_label.configure(image=tk_img)

    # Run again quickly (30ms → ~30 FPS)
    root.after(30, update_frame)


# Start loop
update_frame()
root.mainloop()
cap.release()

