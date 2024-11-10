import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import speech_recognition as sr
import threading

# Initialize MediaPipe Face Mesh with optimized settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3  # Reduced for better performance
)

# Screen setup
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = True

# Set cursor to middle of screen at startup
pyautogui.moveTo(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)

def get_eye_position(landmarks, frame):
    # Using more stable landmarks for eye tracking
    left_eye = np.mean([(landmarks[145].x, landmarks[145].y), 
                        (landmarks[159].x, landmarks[159].y)], axis=0)
    right_eye = np.mean([(landmarks[374].x, landmarks[374].y), 
                         (landmarks[386].x, landmarks[386].y)], axis=0)
    
    # Calculate center point between eyes
    center_point = np.mean([left_eye, right_eye], axis=0)
    
    # Convert to screen coordinates with adjusted range
    screen_x = np.interp(center_point[0], [0.4, 0.6], [0, SCREEN_WIDTH])
    screen_y = np.interp(center_point[1], [0.4, 0.6], [0, SCREEN_HEIGHT])
    
    return screen_x, screen_y

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Smoothing parameters
smoothing = 0.2  # Faster response
last_x, last_y = SCREEN_WIDTH/2, SCREEN_HEIGHT/2

# Add at the start of your code
def listen_for_commands():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                
                # Click commands
                if any(word in command for word in ["click", "tap", "press", "left"]):
                    pyautogui.click()
                
                # Enter key
                elif "enter" in command:
                    pyautogui.press('enter')
                
                # Double click
                elif "double" in command:
                    pyautogui.doubleClick()
                
                # Right click
                elif "right" in command:
                    pyautogui.rightClick()
                
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass

# Add after your screen setup and before the main loop
# Start the voice recognition in a separate thread
command_thread = threading.Thread(target=listen_for_commands, daemon=True)
command_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Process frame
        frame = cv2.flip(frame, 1)  # Mirror flip for more intuitive movement
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            screen_x, screen_y = get_eye_position(landmarks, frame)
            
            # Smooth movement
            smooth_x = last_x + (screen_x - last_x) * smoothing
            smooth_y = last_y + (screen_y - last_y) * smoothing
            
            # Move cursor
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            last_x, last_y = smooth_x, smooth_y

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
