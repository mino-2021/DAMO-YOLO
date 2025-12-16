from moviepy import VideoFileClip
import cv2
import numpy as np

# Load video
clip = VideoFileClip("./demo/snakeboard.mp4")
clip_resized = clip.resized(height=420)

print(f"Video duration: {clip.duration} seconds")
print(f"Original size: {clip.size}")
print(f"Resized size: {clip_resized.size}")
print(f"FPS: {clip.fps}")
print("\nPress 'q' to quit")

# Play video using OpenCV
for frame in clip_resized.iter_frames(fps=clip.fps):
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('DAMO-YOLO Result', frame_bgr)

    # Wait for appropriate time based on FPS (in milliseconds)
    if cv2.waitKey(int(1000/clip.fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
clip.close()
