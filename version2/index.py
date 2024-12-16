import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import time
import numpy as np
import math
import pyrealsense2 as rs
import statistics
import socket

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# profile = pipeline.start(config)
# # depth_sensor = profile.get_device().first_depth_sensor()
# # depth_scale = depth_sensor.get_depth_scale()
# # align = rs.align(rs.stream.color)
# # frames = pipeline.wait_for_frames()
# # aligned_frames = align.process(frames)
# simulate cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


#TKINTER
root = tk.Tk()
root.title("Operational GUI")
running = True
listbox = tk.Listbox(root)
listbox.pack()

video_label = tk.Label(root)
video_label.pack()
def close_app():
    global running
    running = False
    root.destroy()
while running:
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # color_image = np.asanyarray(color_frame.get_data())
    # # Convert the frame to RGB for Tkinter
    # frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # simulator
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    # Display the video feed in the GUI
    video_label.config(image=frame_tk)
    video_label.image = frame_tk
    root.update()

cap.stop()
cv2.destroyAllWindows()