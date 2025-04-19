from flask import Flask, render_template, request, send_file, url_for
import torch
import cv2
import numpy as np
import os
import playsound

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True)

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('index.html')  # Render the index page

@app.route('/upload', methods=['POST'])
def upload():
    # Check if video part is in the request
    if 'video' not in request.files:
        return "No file part", 400
    
    file = request.files['video']
    # Check if filename is empty
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded video to a temporary path
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)

    # Process the video and check for violence
    processed_video_path, violence_detected, violence_percentage = process_video(video_path)

    # Return the result page with the video and the result
    return render_template('result.html', 
                           violence_detected=violence_detected, 
                           violence_percentage=violence_percentage, 
                           processed_video=os.path.basename(processed_video_path))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return video_path, False, 0

    total_frames = 0
    violent_frames = 0
    prev_positions = []
    alarm_triggered = False

    # Save the processed video
    processed_video_path = os.path.join('uploads', 'processed_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving the video
    out = cv2.VideoWriter(processed_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        results = model(frame)

        current_positions = []
        for *box, conf, cls in results.xyxy[0]:
            if model.names[int(cls)] == 'person':
                x1, y1, x2, y2 = map(int, box)
                current_positions.append((x1, y1, x2, y2))
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check for violence detection conditions
        if detect_violence(current_positions) or analyze_motion(prev_positions, current_positions):
            violent_frames += 1
            # Draw circle around person to indicate violence
            for person in current_positions:
                x1, y1, x2, y2 = person
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Circle in the center
                cv2.circle(frame, (cx, cy), 50, (0, 0, 255), 3)
            cv2.putText(frame, "Violence Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Play alarm only once when violence is first detected
            if not alarm_triggered:
                alarm_path = os.path.join('static', 'alarm.mp3')  # Using 'static' folder for alarm.mp3
                if os.path.exists(alarm_path):
                    playsound.playsound(alarm_path)
                alarm_triggered = True
        
        prev_positions = current_positions
        out.write(frame) 

    cap.release()
    out.release()

  
    violence_percentage = (violent_frames / total_frames) * 100 if total_frames > 0 else 0
    violence_detected = violence_percentage > 0  # Violence detected if any violent frame is found
    return processed_video_path, violence_detected, violence_percentage

def detect_violence(current_positions):
  
    for i in range(len(current_positions)):
        for j in range(i + 1, len(current_positions)):
            distance = np.linalg.norm(np.array(current_positions[i][:2]) - np.array(current_positions[j][:2]))
            if distance < 50: 
                return True
    return False

def analyze_motion(prev_positions, current_positions):
   
    for prev, curr in zip(prev_positions, current_positions):
        if np.linalg.norm(np.array(prev[:2]) - np.array(curr[:2])) > 30:  # Threshold for aggressive movement
            return True
    return False

@app.route('/processed_video/<filename>')
def processed_video(filename):
 
    video_path = os.path.join('uploads', filename)
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
