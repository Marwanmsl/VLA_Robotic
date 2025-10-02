import os
import cv2
from google import genai
from google.genai import types
from PIL import Image
import json

# Set your API key
os.environ["GOOGLE_API_KEY"] = "APIkey"

# Initialize client
client = genai.Client()

MODEL_ID = "gemini-robotics-er-1.5-preview"
PROMPT = """
You are an object and scene understanding system.
Return only a JSON response in the format:
{
  "objects": ["object1", "object2", ...],
  "scenario": "Short description of what is happening"
}
"""

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (800, int(800 * height / width)))

    # Convert to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # Call GenAI model
    objects, scenario = [], ""
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
                objects = data.get("objects", [])
                scenario = data.get("scenario", "")
            except json.JSONDecodeError:
                scenario = "ParseError"

    except Exception as e:
        scenario = f"Error: {str(e)}"

    # Draw objects on frame (top-left)
    y_offset = 30
    for obj in objects:
        cv2.putText(frame, f"Object: {obj}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(obj)
        y_offset += 30

    # Draw scenario at bottom
    if scenario:
        cv2.putText(frame, f"Scenario: {scenario}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        print(scenario)

    # Show the live frame
    cv2.imshow("Live Camera", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

