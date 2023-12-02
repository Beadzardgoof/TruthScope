import os
import cv2
import pandas as pd
import numpy as np




# Returns a list containing the path to each video and its category.

def get_videos():
    base_path = "../Datasets/Real Life Trial Cases Data/Clips"
    categories = ["Deceptive", "Truthful"]
    videos = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            videos.append((video_path, category))
    return videos
    

# Preprocesses a single video
##1 Sample video into same number of frames
##2 Crop the face from each frame video
##3 Resize all frames to a similar size
##4 Apply Eulerian magnification algorithm on the frames
def preprocess_video(video_path, eulerian_magnify = False, num_sampled_frames= 300, frame_size = [128,128]):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(total_frames)
    # Calculate the interval at which to sample frames
    frame_sample_rate = max(1, total_frames // num_sampled_frames)

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # List to store processed frames
    processed_frames = []
    # Counter for the current frame in the video
    current_frame = 0

    # Process the video until it ends
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Break the loop if there are no more frames
        if not ret:
            break

        # Process frames at intervals determined by frame_sample_rate
        if current_frame % frame_sample_rate == 0:
            # Convert frame to grayscale - necessary for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Process each face found
            for (x, y, w, h) in faces:
                # Crop the face from the frame
                face_frame = frame[y:y+h, x:x+w]
                # Resize the cropped face to a standard size
                resized_face = cv2.resize(face_frame, (frame_size[0], frame_size[1]))

                # Optionally apply Eulerian magnification
                if eulerian_magnify:
                    magnified_frame = apply_eulerian_magnification(resized_face)
                    processed_frames.append(magnified_frame)
                else:
                    # Append the processed face to the list
                    processed_frames.append(resized_face)

            # Stop processing if the desired number of frames has been reached
            if len(processed_frames) >= num_sampled_frames:
                break

        # Increment the frame counter
        current_frame += 1

    # Release the video capture object
    cap.release()
    # Return the list of processed frames
    return processed_frames

# Eulerian mangification algorithm by MIT, it magnifies colors of videos to reveal tiny changes that may hold valuable information.
def apply_eulerian_magnification(frame):
    # Implement the Eulerian magnification algorithm here
    # This part is left as a placeholder for the actual implementation
    pass

# Takes processed frames and saves them as images in output path (for visualization)
def save_frames_as_images(processed_frames, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each frame as an image
    for i, frame in enumerate(processed_frames):
        # Define the filename for each image
        filename = os.path.join(output_folder, f'frame_{i:04d}.png')  # Saves as frame_0001.png, frame_0002.png, etc.
        # Write the image
        cv2.imwrite(filename, frame)




#processed_frames = preprocess_video("BM003_1PT.wmv")
processed_frames = preprocess_video("trial_lie_001.mp4")

save_frames_as_images(processed_frames, "Test")