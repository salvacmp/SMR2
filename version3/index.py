import tkinter as tk
from tkinter import Label, Frame, Entry, Button
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import threading
import time
import random
import socket
import datetime
import struct
import pyrealsense2 as rs
import numpy as np
import math
from eventemitter import EventEmitter
import json

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control")

        # Create main layout with 2 columns
        self.left_frame = Frame(root, width=500, height=500, bg="black")
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.right_top_frame = Frame(root, width=500, height=250, bg="gray")
        self.right_top_frame.grid(row=0, column=1, padx=10, pady=10)

        self.right_bottom_frame = Frame(root, width=500, height=250, bg="white")
        self.right_bottom_frame.grid(row=1, column=1, padx=10, pady=10)

         # Setting up styles for buttons
        self.style = ttk.Style()
      
        # Use the 'alt' theme for a more modern look
        self.style.theme_use('alt')  

        # Style for buttons
        self.style.configure("TButton",
                             padding=6,
                             relief="flat",  # Flat button for a more modern effect
                             background="#4CAF50",  # Nice green background color
                             foreground="white",  # White text
                             font=("Arial", 12, "bold"))

        # Hover effects for buttons
        self.style.map("TButton",
                       background=[("active", "#45a049")])  # Lighter green color when hovering
        
        # New style for the "Stop" button (red background)
        self.style.configure("RedButton.TButton",
                             padding=6,
                             relief="flat",  # Flat button without 3D effect
                             background="#FF5733",  # Red for the Stop button
                             foreground="white",  # White text
                             font=("Arial", 12, "bold"))
       
        # Hover effect for the "Stop" button
        self.style.map("RedButton.TButton",
                      background=[("active", "#FF2A00")])  # Darker red when hovering

        # Ensure that all columns have the same 'weight' so they scale evenly
        for col in range(4):
            self.right_top_frame.grid_columnconfigure(col, weight=1)

        # Ensure that the rows have the correct 'weight'
        self.right_top_frame.grid_rowconfigure(0, weight=1)
        self.right_top_frame.grid_rowconfigure(1, weight=1)
        self.right_top_frame.grid_rowconfigure(2, weight=1)
        self.right_top_frame.grid_rowconfigure(3, weight=1)
        self.right_top_frame.grid_rowconfigure(4, weight=1)
        # Add a placeholder logo in the middle top center
        # self.logo_label = Label(root, text="Ropax Depaletizer", bg="white", font=("Arial", 24))
        # self.logo_label.place(relx=0.5, y=30, anchor="center")
        self.logo_image = ImageTk.PhotoImage(file="logo.png")  # Ensure logo.png is in the same directory
        self.logo_label = Label(root, image=self.logo_image, bg="white")
        self.logo_label.place(relx=0.5, y=30, anchor="center")

        # Live camera feed setup
        self.camera_label = Label(self.left_frame)
        self.camera_label.pack()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # align = rs.align(rs.stream.color)
        # frames = pipeline.wait_for_frames()
        # aligned_frames = align.process(frames)
        # self.cap = cv2.VideoCapture(0)  # Access webcam
        self.emitter = EventEmitter()

        #Detection config
        self.model_path = "E:/1_SMR2/SMR2/UR Script/vision/best.pt"
        self.model = YOLO(self.model_path)
        # self.eye_points = np.float32([
        #     (153, 415),  # Bottom-left (BL)
        #     (405, 391),  # Bottom-right (BR)
        #     (407, 212),  # Top-right (TR)
        #     (213, 213),  # Top-left (TL)
        # ])
        # self.eye_points = np.float32([
        #     (181, 440),
        #     (424, 411),
        #     (424, 232),
        #     (240, 234)
        # ])
        # self.hand_points = np.float32([
        #     (-0.47883,-0.22280),  # Bottom-left (BL)
        #     (-0.42975,-0.68798),  # Bottom-right (BR)
        #     (-0.08388,-0.69761),  # Top-right (TR)
        #     (-0.08238,-0.33500),  # Top-left (TL)
        # ])
        self.eye_points = np.float32([
            (196, 463),
            (533, 462),
            (513, 37),
            (182, 45)
        ])
        self.hand_points = np.float32([
            (-0.04831,-0.36217),  # Bottom-left (BL)
            (-0.04092, -0.74408),  # Bottom-right (BR)
            (-0.52507,-0.38959),  # Top-right (TR)
            (-0.51195, -0.76239),  # Top-left (TL)
        ])
        # self.roi_polygon = [(153, 415), (405, 391), (407, 212), (213, 213)]
        self.roi_polygon = [(267, 468), (463, 462), (448, 216), (254, 219)]

        self.perspective_matrix = cv2.getPerspectiveTransform(self.eye_points, self.hand_points)
        # self.camera_matrix = np.array([[1.07547194e+03, 0.00000000e+00, 3.04854337e+02],
        #                   [0.00000000e+00, 1.07168396e+03, 2.15872880e+02],
        #                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.camera_matrix = np.array([[596.8016357421875, 0, 312.9718017578125],
                          [0, 596.8016357421875, 236.02671813964844],
                          [0.0, 0.0, 1.0]])
        # self.dist_coeffs = np.array([[-3.00103243e-01, 1.53284927e+01, 1.00493178e-02, -8.89411997e-03, -9.81515375e-01]])
        self.dist_coeffs = np.zeros(5)
        #Points Memory
        self.points = []
        self.PointRequest = False
        
        # Placeholder on top right
        # self.placeholder_label = Label(self.right_top_frame, text="Placeholder", font=("Arial", 16), bg="gray", fg="white")
        # self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
                # Top-right frame: IP and Port inputs with Connect button
        self.ip_label = Label(self.right_top_frame, text="IP:", font=("Arial", 12), bg="gray", fg="white")
        self.ip_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.ip_entry = Entry(self.right_top_frame, width=20)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.port_label = Label(self.right_top_frame, text="Port:", font=("Arial", 12), bg="gray", fg="white")
        self.port_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.port_entry = Entry(self.right_top_frame, width=20)
        self.port_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Buttons for Connect and Disconnect
        self.connect_button = ttk.Button(self.right_top_frame, text="Connect", command=self.connect)
        self.connect_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.disconnect_button = ttk.Button(self.right_top_frame, text="Disconnect", command=self.disconnect)
        self.disconnect_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # Buttons for Start, Pause
        self.start_button = ttk.Button(self.right_top_frame, text="Start", command=self.start_process, state="disabled")
        self.start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.pause_button = ttk.Button(self.right_top_frame, text="Pause", command=self.pause_process, state="disabled")
        self.pause_button.grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # Stop button (spanning 2 columns)
        self.stop_button = ttk.Button(self.right_top_frame, text="Stop", command=self.stop_process, state="disabled", style="RedButton.TButton")
        self.stop_button.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        # Logs section
        self.log_listbox = tk.Listbox(self.right_bottom_frame, height=15, width=150)
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        self.log_scrollbar = tk.Scrollbar(self.right_bottom_frame)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_listbox.config(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.config(command=self.log_listbox.yview)

        # Load IP and Port from JSON
        self.load_config()

        # Socket server setup
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 2025))
        self.server_socket.listen(5)
        
        # Robot Socket
        self.robot_socket = None
        

        # Start threads for camera and log updates
        self.running = True
        threading.Thread(target=self.update_camera_feed, daemon=True).start()
        threading.Thread(target=self.socket_server, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    def load_config(self):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                self.ip_entry.insert(0, config.get("ip", ""))
                self.port_entry.insert(0, config.get("port", ""))
        except FileNotFoundError:
            with open("config.json", "w") as file:
                json.dump({"ip": "", "port": ""}, file)

    def save_config(self):
        config = {
            "ip": self.ip_entry.get(),
            "port": self.port_entry.get()
        }
        with open("config.json", "w") as file:
            json.dump(config, file)

    def connect(self):
        self.save_config()
        # Simulate connection logic here
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            
            self.robot_socket.connect((self.ip_entry.get(), int(self.port_entry.get())))
            # self.robot_socket.setblocking(False)
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            self.insertLog("Connected to IP: {} Port: {}".format(self.ip_entry.get(), self.port_entry.get()))
            self.insertLog(f"{str(data)}")

            # self.insertLog(f"{str(data)}")
            if data == "Connected: Universal Robots Dashboard Server":
                load_program = "load /programs/ropax/main.urp\n"
                self.robot_socket.send(load_program.encode())
                data = self.robot_socket.recv(1024)
                data = data.decode('utf-8').strip()
                self.insertLog(f"{data}")
                self.active = True
                self.update_button_states()
                # self.robot_socket.close()
            else:
                self.insertLog("Failed to connect to the robot.")
                self.robot_socket.close()
                self.active = False
                self.update_button_states()
        except Exception as e:
            self.insertLog(f"Error: {e}")
            self.active = False
            self.update_button_states()
    def disconnect(self):
        self.robot_socket.close()
        self.insertLog("Disconnected from IP: {} Port: {}".format(self.ip_entry.get(), self.port_entry.get()))
        self.active = False
        self.update_button_states()

    def update_button_states(self):
        state = "normal" if self.active else "disabled"
        self.start_button.config(state=state)
        self.pause_button.config(state=state)
        self.stop_button.config(state=state)
    
    def start_process(self):
        # self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.robot_socket.connect((self.ip_entry.get(), int(self.port_entry.get())))
        # data = self.robot_socket.recv(1024)
        # data = data.decode('utf-8').strip()
        try:
           
            start_program = "play\n"
            self.robot_socket.sendall(start_program.encode())
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            
            self.insertLog(f"{data}")
            self.insertLog("Process started.")
        except Exception as e:
            self.insertLog(f"Error: {e}")

    def pause_process(self):
        self.log_listbox.insert(tk.END, "Process paused.")

    def stop_process(self):
        self.log_listbox.insert(tk.END, "Process stopped.")
        self.active = False
        self.update_button_states()
    def transform_coordinates(self, xp, yp, zp):
        a = -7.623178493625794e-05
        b = 0.001583300304299284
        c = -0.5841489756392849
        d = 0.0015555086433522267
        e = 6.071404193758092e-05
        f = -1.115529503523589

        xd = a * xp + b * yp + c
        yd = d * xp + e * yp + f

        return xd, yd
    def update_camera_feed(self):
        while self.running:
            
            # Convert frame to RGB and then to ImageTk
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame:
                continue
            # Convert the frame to RGB for Tkinter
            h, w = color_image.shape[:2]
            new_camera_matrix, undist_roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            undistorted_image = cv2.undistort(color_image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            x, y, w, h = undist_roi
            undistorted_image = undistorted_image[y:y+h, x:x+w]
            
            depth_image = np.asanyarray(depth_frame.get_data())
            undistorted_depth = cv2.undistort(depth_image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            
            # new methods
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            # h, w = color_image.shape[:2]
            # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            # map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
            # undistorted_color = cv2.remap(color_image, map1, map2, interpolation=cv2.INTER_LINEAR)

            # Undistort depth image
            # map1_depth, map2_depth = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
            # undistorted_depth = cv2.remap(depth_image, map1_depth, map2_depth, interpolation=cv2.INTER_NEAREST)
            
            # No Undistortion
            
            
            # frame = color_image
            frame = undistorted_image
            frame_height, frame_width = frame.shape[:2]
            results = self.model(frame)
            annotated_frame = frame.copy()
            camera_z1 = 1.086
            robot_z1 = 0.25009
            camera_z2 = 1.132
            robot_z2 = 0.20011
            slope, intercept = self.calculate_slope_and_intercept(camera_z1, robot_z1, camera_z2, robot_z2)
            annotated_frame = results[0].plot()
            #Center Calculation
            cv2.circle(annotated_frame, (320,240), 5, (0, 0, 255), -1)
            # #YOLO
            # cv2.circle(annotated_frame, (196, 463), 5, (0, 0, 255), -1)
            # cv2.circle(annotated_frame, (533, 462), 5, (0, 0, 255), -1)
            # cv2.circle(annotated_frame, (519, 452), 5, (0, 0, 255), -1)
            # cv2.circle(annotated_frame, (180, 459), 5, (0, 0, 255), -1)
            
            cv2.circle(annotated_frame, (498, 227), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (495, 30), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (198, 38), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (207, 248), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (221, 444), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (506, 432), 5, (0, 0, 255), -1)
            
            
            obbs = results[0].obb
            if obbs is not None:
                print(len(obbs))
                for obb in obbs.data.cpu().numpy():
                    # Extract OBB details
                    x_center, y_center, width, height, angle, confidence, class_id = obb
                    #detection square
                    if confidence >= 0.55:
                        if True:
                            corners = self.get_obb_corners(x_center, y_center, width, height, angle)
                            cornerList = []

                            # Collect y-coordinates along with their index
                            for i, corner in enumerate(corners):
                                cornerList.append((corner[1], i))

                            # Sort the corners based on their y-coordinates (ascending)
                            sorted_corners = sorted(cornerList, key=lambda x: x[0])

                            # Get the three bottom-most corners
                            bottom_three = sorted_corners[-3:]

                            # Extract the most bottom point
                            bottom_point_index = bottom_three[-1][1]
                            bottom_point = corners[bottom_point_index]

                            # Draw the bottom-most point
                            cv2.circle(annotated_frame, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 0, 0), -1)

                            # Calculate distances and store the points
                            distances = []
                            for y_coord, index in bottom_three[:-1]:  # Skip the bottom-most point
                                current_point = corners[index]

                                # Calculate the distance
                                distance = np.sqrt((current_point[0] - bottom_point[0]) ** 2 + (current_point[1] - bottom_point[1]) ** 2)
                                distances.append((distance, current_point))

                            # Sort distances to get the longest side
                            distances.sort(reverse=True, key=lambda x: x[0])
                            longest_side_distance, longest_side_point = distances[0]

                            # Calculate the angle for the longest side
                            dx = longest_side_point[0] - bottom_point[0]
                            dy = longest_side_point[1] - bottom_point[1]
                            angle = math.degrees(math.atan2(dy, dx))  # Angle in degrees
                            if angle < 0: angle *=-1
                            # Display the longest side and its angle
                            cv2.line(
                                annotated_frame,
                                (int(bottom_point[0]), int(bottom_point[1])),
                                (int(longest_side_point[0]), int(longest_side_point[1])),
                                (0, 255, 0),
                                2
                            )
                            # cv2.putText(
                            #     annotated_frame,
                            #     f"Angle: {angle:.2f}°",
                            #     (int((longest_side_point[0] + bottom_point[0]) / 2), int((longest_side_point[1] + bottom_point[1]) / 2) - 10),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (255, 255, 0),
                            #     1
                            # )

                            # Draw a horizontal line at the bottom-most point’s y-coordinate
                            cv2.line(
                                annotated_frame,
                                (0, int(bottom_point[1])),
                                (frame_width, int(bottom_point[1])),
                                (0, 0, 255),
                                2
                            )

                            
                        # Simulate coordinates with the origin at the bottom-left
                            hand_x, hand_y = self.map_to_hand_plane((x_center, y_center), self.perspective_matrix)
                            brick_rotation = self.calculate_brick_rotation(width, height, angle)

                            # print(repr(brick_rotation))
                            cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                            
                            cv2.line(annotated_frame, (int(x_center), int(y_center)), (int(x_center), 10), (0, 255, 0), 2)
                            # cv2.putText(
                            #     annotated_frame,
                            #     f"Rotation: {int(np.degrees(angle) % 360)}° | {angleOffset(int(np.degrees(angle)% 360))}° | NC: {radians_to_degrees(angle) }",
                            #     (int(x_center) + 10, int(y_center) - 30),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (255, 0, 0),
                            #     1
                            # )
                            # Display the simulated coordinates and class name
                            class_name = results[0].names[int(class_id)]
                            print(class_name)
                            # cv2.putText(
                            #     annotated_frame,
                            #     f"Hand: ({hand_x:.3f}, {hand_y:.3f})",
                            #     (int(x_center) + 10, int(y_center) - 10),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (0, 255, 0),
                            #     1
                            # )
                            depth_value = depth_frame.get_distance(int(x_center), int(y_center)) / depth_scale
                            depth_value = depth_value / 1000
                            # depth_value = undistorted_depth[int(x_center), int(y_center)] * depth_scale
                            counted_Z = self.camera_to_robot_z(depth_value, slope, intercept) -0.005
                            cx, cy = self.transform_coordinates(x_center, y_center, 1)
                            #calculate offset
                            ox, oy = self.calculate_distance(cx, cy, depth_value*1000)
                            
                            cv2.putText(
                                annotated_frame,
                                f"Z: {ox:.5f} | o_y: {oy:.5f}",
                                (int((longest_side_point[0] + bottom_point[0]) / 2), int((longest_side_point[1] + bottom_point[1]) / 2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1
                            )
                            if class_name == "brick":
                                class_type = 1
                            if class_name == "brick-side":
                                class_type = 2   
                            # print(repr(counted_Z))
                            if hand_x is not None and self.PointRequest and len(self.points) <= len(obbs):
                                angle_converted = angle
                                # angle_counted = angleOffset(angle_converted)
                                angle_counted = self.newAngleCounter(angle_converted)
                                self.insertLog(f"Real angle: {angle_converted} | Offset Angle: {angle_counted} | CamZ : {depth_value} | HandZ: {counted_Z}")
                                self.insertLog([cx ,cy, depth_value, int( angle_counted), class_name])
                                self.points.append([cx,cy, depth_value, int(angle_counted), class_type])
                                if len(self.points) == len(obbs):
                                    self.PointRequest = False 
                                
                                
                        
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            time.sleep(0.03)  # 30 FPS
    def insertLog(self, message):
        self.log_listbox.insert(0, f"{datetime.datetime.now()} | {message}")
    def socket_server(self):
        while self.running:
            # self.log_listbox.insert(tk.END, "Socket Server Started")
            self.insertLog("Socket Server Started")
            while True:
                client_socket, client_address = self.server_socket.accept()
                # self.log_listbox.insert(tk.END, f"Connection established with {client_address}")
                self.insertLog(f"Connection established with {client_address}")
                try:
                    while True:
                        data = client_socket.recv(1024)  # Buffer size
                        # print(repr(data))
                        if not data:
                            self.insertLog(f"Connection closed by {client_address}")
                            # print(f"Connection closed by {client_address}")
                            break
                        recv = struct.unpack('>I', data)[0]
                        self.insertLog(f"Received: {recv}")
                        if recv == 99991: #set Point
                            self.points = []
                            self.PointRequest = True
                            while self.PointRequest:
                                time.sleep(0.1)
                            client_socket.send(str(f"({len(self.points)})").encode())
                            # def sendReturn():
                            #     client_socket.send(str(f"({len(self.points)})").encode())
                            #     self.PointRequest = False
                        # client_socket.send(str("OK").encode())
                        if recv < 99991:
                            if len(self.points) == 0:
                                client_socket.sendall(str(f"()").encode())
                                break
                            index = recv
                            print(f"Request points: {index}")
                            print(repr(self.points))
                            points = self.points[int(index)]
                            print(f"x: {points[0]}, y: {points[1]}, height: {points[2]}, radians: {math.radians(points[3])}, type: {points[4]}")
                            client_socket.sendall(str(f"({points[0]}, {points[1]}, {(points[2])}, {(points[3])}, {(points[4])})").encode())    
                except Exception as e:
                    self.insertLog(f"Error: {e}")

                finally:
                    # Clean up the connection
                    client_socket.close()


    def calculate_offset_xy(self, detected_points, depth):
        """
        Calculate the offset of the detected object from the center of the camera based on the depth,
        considering only the x and y axes.
        
        :param detected_points: 2D points of detected object in the image plane (x, y)
        :param depth: Depth value of the object in camera space (in meters)
        :return: Offset in 2D space (dx, dy)
        """
        # Uncomment this if you need to undistort the detected 2D points
        # undistorted_points = cv2.undistortPoints(detected_points, self.camera_matrix, self.dist_coeffs)
        
        # Use detected_points directly if undistortion is not needed
        points = detected_points  # or undistorted_points if undistortion is used

        # Calculate the 3D coordinates in camera space using depth (z)
        x = (points[0][0] - self.camera_matrix[0][2]) * depth / self.camera_matrix[0][0]
        y = (points[0][1] - self.camera_matrix[1][2]) * depth / self.camera_matrix[1][1]
        
        # Calculate the offset in 2D space (dx, dy)
        dx = x
        dy = y

        return dx, dy


    def calculate_distance(self,pixel_x, pixel_y, distance_mm):
        # Convert pixel coordinates to normalized image coordinates
        normalized_x = (pixel_x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        normalized_y = (pixel_y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]

        # Calculate real-world coordinates
        real_x = normalized_x * distance_mm
        real_y = normalized_y * distance_mm

        return real_x, real_y

    def is_point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using cv2.pointPolygonTest.
        :param point: Tuple (x, y) representing the point.
        :param polygon: List of points [(x1, y1), (x2, y2), ...] representing the polygon.
        :return: True if the point is inside the polygon, False otherwise.
        """
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0
    def map_to_hand_plane(self, point, matrix):
        """
        Maps a point from the eye plane to the hand plane using the perspective matrix.
        :param point: (x, y) tuple in the eye plane.
        :param matrix: Perspective transformation matrix.
        :return: (x, y) tuple in the hand plane.
        """
        src_point = np.array([[point]], dtype=np.float32)  # Shape: (1, 1, 2)
        dst_point = cv2.perspectiveTransform(src_point, matrix)  # Shape: (1, 1, 2)
        return dst_point[0, 0]  # Extract (x, y)

    def angleOffset(self, Angle):
        if Angle < 0 :
            Angle * -1
        if Angle >= 85 and Angle <= 180 :
            return Angle #OK
        if Angle <85 and Angle >=60:
            return Angle + 90 #OK
        if Angle < 60 and Angle >= 0:
            return 180-Angle 
    def get_obb_corners(self, x_center, y_center, width, height, angle):
        """
        Calculate the four corners of an Oriented Bounding Box (OBB).
        :param x_center: X-coordinate of the OBB center.
        :param y_center: Y-coordinate of the OBB center.
        :param width: Width of the OBB.
        :param height: Height of the OBB.
        :param angle: Rotation angle of the OBB in radians.
        :return: List of (x, y) tuples for the four corners.
        """
        # Calculate half dimensions
        half_width = width / 2
        half_height = height / 2
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Define the corner points relative to the center
        corners = np.array([
            [-half_width, -half_height],  # Bottom-left
            [ half_width, -half_height],  # Bottom-right
            [ half_width,  half_height],  # Top-right
            [-half_width,  half_height],  # Top-left
        ])
        
        # Rotate and translate the corners to the global coordinate system
        rotated_corners = np.dot(corners, rotation_matrix.T) + [x_center, y_center]
        
        return rotated_corners
    def calculate_brick_rotation(self, width, height, angle):
        """
        Calculate the brick's rotation relative to the bottom line.
        :param width: Width of the OBB.
        :param height: Height of the OBB.
        :param angle: Orientation angle of the OBB in degrees.
        :return: Adjusted angle (0° for short side down, 90° for long side down).
        """
        # Normalize the angle to the range [0, 180]
        normalized_angle = angle % 180

        if width < height:  # Short side pointing down
            rotation = normalized_angle
        else:  # Long side pointing down
            rotation = (normalized_angle + 90) % 180

        return rotation
    def newAngleCounter(self, Angle):
        if Angle < 65:
            return Angle + 180
        else:
            return Angle
        
    def calculate_slope_and_intercept(self, camera_z1, robot_z1, camera_z2, robot_z2):
        """
        Calculate the slope and intercept for the linear relationship between camera and robot Z-axis.

        Parameters:
            camera_z1 (float): First camera Z-axis value.
            robot_z1 (float): First robot Z-axis value.
            camera_z2 (float): Second camera Z-axis value.
            robot_z2 (float): Second robot Z-axis value.

        Returns:
            tuple: (slope, intercept)
        """
        slope = (robot_z2 - robot_z1) / (camera_z2 - camera_z1)
        intercept = robot_z1 - slope * camera_z1
        return slope, intercept
    def camera_to_robot_z(self, camera_z, slope, intercept):
        """
        Convert camera Z-axis value to robot Z-axis value using the slope and intercept.

        Parameters:
            camera_z (float): Camera Z-axis value.
            slope (float): Slope of the linear relationship.
            intercept (float): Intercept of the linear relationship.

        Returns:
            float: Corresponding robot Z-axis value.
        """
        return slope * camera_z + intercept
    def on_closing(self):
        self.running = False
        self.pipeline.stop()
        self.root.destroy()
        self.server_socket.close()  # Add this line


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
