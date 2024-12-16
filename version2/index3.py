import tkinter as tk
from tkinter import Label, Frame, Entry, Button
import cv2
from PIL import Image, ImageTk
import threading
import time
import random
import json

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter Camera and Logs Interface")

        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Create main layout with 2 columns
        self.left_frame = Frame(root, bg="black")
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.right_top_frame = Frame(root, bg="gray")
        self.right_top_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.right_bottom_frame = Frame(root, bg="white")
        self.right_bottom_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Add a logo image in the middle top center
        self.logo_image = ImageTk.PhotoImage(file="logo.png")  # Ensure logo.png is in the same directory
        self.logo_label = Label(root, image=self.logo_image, bg="white")
        self.logo_label.place(relx=0.5, y=30, anchor="center")

        # Live camera feed setup
        self.camera_label = Label(self.left_frame)
        self.camera_label.pack(fill="both", expand=True)
        self.cap = cv2.VideoCapture(0)  # Access webcam

        # Top-right frame: IP and Port inputs with Connect button
        self.ip_label = Label(self.right_top_frame, text="IP:", font=("Arial", 12), bg="gray", fg="white")
        self.ip_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.ip_entry = Entry(self.right_top_frame, width=20)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.port_label = Label(self.right_top_frame, text="Port:", font=("Arial", 12), bg="gray", fg="white")
        self.port_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.port_entry = Entry(self.right_top_frame, width=20)
        self.port_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.connect_button = Button(self.right_top_frame, text="Connect", command=self.connect)
        self.connect_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Buttons for Start, Pause, Stop
        self.start_button = Button(self.right_top_frame, text="Start", command=self.start_process, state="disabled")
        self.start_button.grid(row=3, column=0, padx=5, pady=5)

        self.pause_button = Button(self.right_top_frame, text="Pause", command=self.pause_process, state="disabled")
        self.pause_button.grid(row=3, column=1, padx=5, pady=5)

        self.stop_button = Button(self.right_top_frame, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Logs section
        self.log_listbox = tk.Listbox(self.right_bottom_frame, height=15)
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=True)
        self.log_scrollbar = tk.Scrollbar(self.right_bottom_frame)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_listbox.config(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.config(command=self.log_listbox.yview)

        # Load IP and Port from JSON
        self.load_config()

        # Start threads for camera and log updates
        self.running = True
        self.active = False
        threading.Thread(target=self.update_camera_feed, daemon=True).start()
        threading.Thread(target=self.generate_logs, daemon=True).start()

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
        self.log_listbox.insert(tk.END, "Connected to IP: {} Port: {}".format(self.ip_entry.get(), self.port_entry.get()))
        self.active = True
        self.update_button_states()

    def update_button_states(self):
        state = "normal" if self.active else "disabled"
        self.start_button.config(state=state)
        self.pause_button.config(state=state)
        self.stop_button.config(state=state)

    def start_process(self):
        self.log_listbox.insert(tk.END, "Process started.")

    def pause_process(self):
        self.log_listbox.insert(tk.END, "Process paused.")

    def stop_process(self):
        self.log_listbox.insert(tk.END, "Process stopped.")
        self.active = False
        self.update_button_states()

    def update_camera_feed(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB and then to ImageTk
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            time.sleep(0.03)  # 30 FPS

    def generate_logs(self):
        while self.running:
            log_message = f"Log entry {random.randint(1000, 9999)}"
            self.log_listbox.insert(tk.END, log_message)
            self.log_listbox.see(tk.END)
            time.sleep(2)  # Add log every 2 seconds

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
