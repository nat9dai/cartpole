import cv2
import os
from tqdm import tqdm

# Path to the folder where frames are saved
frame_folder = 'frames_050/'

frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])

# Define a standard size for the video (e.g., 1280x720)
standard_size = (1280, 720)  # New size (width, height)
frame_rate = 1000  # Test with a lower frame rate

# Initialize the VideoWriter
video = cv2.VideoWriter('050.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, standard_size)

# Write frames to video
for frame_name in tqdm(frames, desc="Creating video", unit="frame"):
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)

    # Resize the frame to the standard size
    if frame is not None:
        frame_resized = cv2.resize(frame, standard_size)
        video.write(frame_resized)
    else:
        print(f"Error loading frame: {frame_name}")

# Release the video writer
video.release()

print("Video has been created successfully!")
