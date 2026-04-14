import time
import socket
import numpy as np
import os
import datetime
import csv

import torch
from PIL import Image
# from torchvision import transforms
import threading
import cv2
import hydra
from omegaconf import OmegaConf
from queue import Queue
import argparse
import signal

try:
    from planner import call_planner
except Exception:
    call_planner = None

global stop_capture, current_frame, trigger_new_instruction
stop_capture = False
current_frame = None
trigger_new_instruction = False
current_step = 1

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global stop_capture
    print("\nSIGINT received. Shutting down...")
    stop_capture = True
    time.sleep(0.5)
    cv2.destroyAllWindows()


signal.signal(signal.SIGINT, signal_handler)

def get_local_ip():
    """Return the current machine's primary LAN IP."""
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No traffic is actually sent for UDP connect; this reveals the outbound interface.
        test_socket.connect(("8.8.8.8", 80))
        return test_socket.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        test_socket.close()


def resolve_server_address(preferred_host, preferred_port, probe_timeout=0.75):
    """
    Prefer the original server address when reachable.
    Otherwise, fall back to this computer's IP.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(probe_timeout)
    try:
        probe.connect((preferred_host, preferred_port))
        print(
            f"Using preferred server {preferred_host}:{preferred_port} "
            "(reachable)."
        )
        return preferred_host, preferred_port
    except OSError:
        local_ip = get_local_ip()
        print(
            f"Preferred server {preferred_host}:{preferred_port} not reachable. "
            f"Falling back to local IP {local_ip}:{preferred_port}."
        )
        return local_ip, preferred_port
    finally:
        probe.close()


class ThreadedCamera(object):
    def __init__(self, src=0, fps=30, max_reopen_attempts=15, reopen_delay=1.0):
        self.src = src
        self.max_reopen_attempts = max_reopen_attempts
        self.reopen_delay = reopen_delay
        self.reopen_attempts = 0
        self.failed = False
        self.failure_reason = None
        print(f"Initializing camera {src}...")
        self.capture = cv2.VideoCapture(src)
        
        if not self.capture.isOpened():
            print(f"Error: Could not open camera {src}")
        else:
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera {src} initialized: {width}x{height}")
        
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        self.FPS = False
        self.FPS_MS = False
        if fps > 0:
            self.FPS = 1/fps
            self.FPS_MS = int(self.FPS * 1000)
            
        self.frame = None
        self.read_once = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera {src} initialized in threaded mode")
        
    def update(self):
        try:
            while not stop_capture:
                if not self.capture.isOpened():
                    self.reopen_attempts += 1
                    if self.reopen_attempts > self.max_reopen_attempts:
                        self.failed = True
                        self.failure_reason = (
                            f"Camera {self.src} did not open after "
                            f"{self.max_reopen_attempts} attempts."
                        )
                        print(self.failure_reason)
                        break
                    print(
                        f"Warning: Camera {self.src} not opened. "
                        f"Retry {self.reopen_attempts}/{self.max_reopen_attempts}..."
                    )
                    self.capture = cv2.VideoCapture(self.src)
                    time.sleep(self.reopen_delay)
                    continue
                    
                (status, tmp_frame) = self.capture.read()
                if status:
                    self.reopen_attempts = 0
                    self.read_once = True
                    self.frame = tmp_frame
                    h, w = self.frame.shape[:2]
                    center_h, center_w = h // 2, w // 2
                    size = min(h, w) // 2
                    self.frame = self.frame[center_h - size:center_h + size, center_w - size:center_w + size]
                    self.frame = cv2.resize(self.frame, (640, 640))
                else:
                    print(f"Warning: Failed to read frame from camera {self.src}")
                    time.sleep(0.1)
                
                if self.FPS: 
                    time.sleep(self.FPS)
        except Exception as e:
            print(f"Error in camera thread for camera {self.src}: {e}")
    
    def read(self):
        if self.read_once and self.frame is not None: 
            return True, self.frame.copy()
        else:
            return False, None
    
    def release(self):
        if self.capture is not None:
            self.capture.release()

def camera_thread_function(
    camera_id,
    global_frame_var,
    fps=30,
    max_reopen_attempts=15,
    camera_init_timeout=8.0,
):
    """Thread function to continuously update the global frame variable with camera feed"""
    global stop_capture
    
    camera = None
    try:
        print(f"Starting camera thread for camera {camera_id}...")
        camera = ThreadedCamera(
            src=camera_id,
            fps=fps,
            max_reopen_attempts=max_reopen_attempts,
        )

        init_start = time.time()
        while not stop_capture and not camera.read_once and not camera.failed:
            if time.time() - init_start > camera_init_timeout:
                camera.failed = True
                camera.failure_reason = (
                    f"Camera {camera_id} did not produce frames within "
                    f"{camera_init_timeout:.1f}s."
                )
                break
            time.sleep(0.1)

        if camera.failed:
            print(
                camera.failure_reason
                or f"Camera {camera_id} failed during initialization."
            )
            stop_capture = True
            return
        
        while not stop_capture:
            ret, frame = camera.read()
            if ret:
                globals()[global_frame_var] = frame
            else:
                time.sleep(0.1)
                continue
            
            time.sleep(0.033)
            
    except Exception as e:
        print(f"Error in camera thread {camera_id}: {e}")
    finally:
        # Ensure camera is released even if there's an exception
        if camera is not None:
            print(f"Releasing camera {camera_id}...")
            camera.release()
        
        print(f"Camera {camera_id} thread exited cleanly")

def extract_square_region(image, target_size=640):
    """
    Extract a square region from the center of an image and resize it.
    
    Args:
        image: Input image (numpy array)
        target_size: Size of the output square image (default: 640)
        
    Returns:
        Square image of target_size x target_size
    """
    # Check if image is None
    if image is None:
        print("Warning: Input image is None")
        return None
    # Get image dimensions
    h, w = image.shape[:2]
    # Calculate center of the image
    center_h, center_w = h // 2, w // 2
    # Calculate the size of the square region to extract
    size = min(h, w) // 2
    # Extract square region
    square_img = image[center_h - size:center_h + size, center_w - size:center_w + size]
    # Resize to target size
    resized_img = cv2.resize(square_img, (target_size, target_size))
    
    return resized_img

class MiniNav:
    def __init__(self, server_address, model_cfg_path, session_dirs=None):
        global current_step
        # Initialize MiniNav parameters
        self.server_address = server_address
        self.plan = [{ "step": 1, "action": "Above", "target": "Red Disk" },
                     { "step": 2, "action": "Toward", "target": "Blue Disk" },
                     { "step": 3, "action": "Turn", "target": "Right 60d" },
                     { "step": 4, "action": "Left", "target": "Man in a suit" }]  # Initial plan with proper formatting
        self.stop_queue = Queue(maxsize=30)
        self.stop_thresh = 0.85
        self.frame_count = 0
        self.fps = 0
        self.running = True
        self.complete = False
        self.client_socket = None  # Initialize socket attribute
        self.yr = 20 * np.pi/180 # degrees per second
        self.step = current_step

        # Use provided session directories if available
        if session_dirs:
            self.session_timestamp = os.path.basename(session_dirs["session_root"])
            self.log_dir = session_dirs["mininav_data_dir"]
            self.images_dir = session_dirs["images_dir"]
        else:
            # Create session timestamp and logging directories (legacy mode)
            self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join("real_mininav", self.session_timestamp)
            self.images_dir = os.path.join(self.log_dir, "images")
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Print logging directories
        print(f"MiniNav logging to: {self.log_dir}")
        print(f"MiniNav images will be saved to: {self.images_dir}")
        print("\n -------------------------------------------------- \n")

        # Connect to the server
        if not self.connect_to_server():
            raise RuntimeError("Failed to connect to the server after multiple attempts")
        print("Connected to server successfully")
        print("\n -------------------------------------------------- \n")

        # Load model using config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.load_model_from_config(model_cfg_path)
        print("Model loaded successfully")
        print("\n -------------------------------------------------- \n")

        # ask for first instruction
        self.plan_n_check()

        # Initialize CSV log file
        self.csv_path = os.path.join(self.log_dir, "flight_data.csv")
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'step_num', 'action', 'target', 
                            'vx', 'vy', 'vz', 'yaw', 'stop_signal', 
                            'step_remain', 'image_path'])
        # Log model config info
        self.log_config_info(model_cfg_path)

    def connect_to_server(self):
        """Establish connection with the control server"""
        con_attempts = 0
        max_attempts = 10  # Set a maximum number of attempts
        
        while con_attempts < max_attempts:
            con_attempts += 1
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect(self.server_address)
                print(f"Connected to server at {self.server_address[0]}:{self.server_address[1]}")
                with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
                    f.write(f"{datetime.datetime.now().isoformat()} - Connected to server at {self.server_address[0]}:{self.server_address[1]}\n")
                return True
            except Exception as e:
                print(f"Connection attempt {con_attempts}/{max_attempts} failed: {e}")
                # wait 5 seconds before trying again
                time.sleep(5)
    
        print("Failed to connect to server after multiple attempts")
        with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - Failed to connect to server after {max_attempts} attempts\n")
        return False
    
    def log_config_info(self, model_cfg_path):
        """Log model configuration information"""
        config_log_path = os.path.join(self.log_dir, "config_info.txt")
        with open(config_log_path, 'w') as f:
            f.write(f"Session started: {self.session_timestamp}\n")
            f.write(f"Model path: {model_cfg_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Stop threshold: {self.stop_thresh}\n")
            f.write(f"Stop queue size: {self.stop_queue.maxsize}\n")
            f.write(f"Server address: {self.server_address}\n\n")
            f.write("Initial plan:\n")
            for step in self.plan:
                f.write(f"  Step {step['step']}: {step['action']} - {step['target']}\n")

    def load_model_from_config(self, cfg_path):
        d, t, c = cfg_path
        cfg_path = os.path.join('/home/choukram/repos/flex/local/train_flight', d, t)
        cfg = OmegaConf.load(os.path.join(cfg_path, "config")).model
        
        # Set up feature extraction flag
        cfg.net.extractor.extract_features_flag = True
        
        # Instantiate the model
        model = hydra.utils.instantiate(cfg)
        
        # Load checkpoint
        ckpt_path_base = os.path.join(cfg_path, "checkpoints")
        if c == -1:
            ckpt_paths = sorted([f for f in os.listdir(ckpt_path_base) if f.endswith('.ckpt')])
            cpath = os.path.join(ckpt_path_base, ckpt_paths[-1])
        else:
            cpath = os.path.join(ckpt_path_base, f'step_{c}.ckpt')

        if not os.path.exists(cpath): 
            raise ValueError(f"Check the checkpoint path {cpath}")
        
        ckpt = torch.load(cpath, map_location=torch.device('cpu'), weights_only=False)
        ckpt = ckpt['state_dict']
        
        # Load model components
        model.net.policy.load_state_dict(ckpt['policy'])
        model.net.extractor.last_linear_layer.load_state_dict(ckpt['extractor_ll'])
        
        if model.net.stop_flagger is not None and "stop_flagger" in ckpt.keys():
            model.net.stop_flagger.load_state_dict(ckpt['stop_flagger'])
            if cfg.net.stop_flagger_cfg.name == 'temporal': 
                model.net.stop_flagger.single_step = True
        
        return model.to(self.device)
    
    def plan_n_check(self):
        global current_step
        if call_planner is None:
            print("Planner unavailable; using the current in-memory plan.")
            return

        while True:
            instruction = input("Enter the new text instruction: ")
            plan = call_planner(instruction)
            if not plan:
                print("⚠️ No steps returned from the planner.")
                continue

            print("Plan received from planner:")
            for step in plan:
                print(f"Step {step['step']}: {step['action']}, {step['target']}")

            confirm = input("Do you want to execute this plan? (y/n): ")
            if confirm.lower() == 'y':
                self.plan = plan
                self.reset_state()
                current_step = 1
                self.log_new_plan(instruction, plan)
                return
            else:
                print("Plan not executed. Waiting for a new instruction.")
    
    def log_new_plan(self, instruction, plan):
        """Log when a new plan is created"""
        plan_log_path = os.path.join(self.log_dir, f"plan_{datetime.datetime.now().strftime('%H%M%S')}.txt")
        with open(plan_log_path, 'w') as f:
            f.write(f"Instruction: {instruction}\n")
            f.write("Plan steps:\n")
            for step in plan:
                f.write(f"  Step {step['step']}: {step['action']} - {step['target']}\n")

    def save_image(self, ts, frame):
        # Create display frame with overlays
        display_frame = frame.copy()
        
        # Add step information
        step_text = f"Step {self.step}: {self.plan[self.step-1]['action']} {self.plan[self.step-1]['target']}"
        cv2.putText(display_frame, step_text, (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add FPS information in the bottom right corner
        fps_text = f"FPS: {self.fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = display_frame.shape[1] - text_size[0] - 20
        text_y = display_frame.shape[0] - 20
        cv2.putText(display_frame, fps_text, (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame 800x800 size
        display_frame = cv2.resize(display_frame, (800, 800))

        # Convert OpenCV frame to PIL image
        cv2_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb)

        img_filename = f"{ts}_step{self.step}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)
        pil_img.save(img_path)

        return img_filename

    def image_to_tensor(self, np_image):
        # Convert from BGR (OpenCV) to RGB
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        img = Image.fromarray(np_image)
        
        # Make the image have 3 channels
        img = img.convert('RGB')

        # Resize the image to 224x224 as in inference_test.py
        img = img.resize((224, 224))

        # Convert PIL image to numpy array
        img_np = np.array(img).astype(np.float32) / 255.0
        # Change from (H, W, C) to (C, H, W) format
        img_np = img_np.transpose((2, 0, 1))
        # Convert to tensor and move to device
        img_tensor = torch.from_numpy(img_np).to(self.device)

        return img_tensor

    def inference(self, np_image):
        img_tensor = self.image_to_tensor(np_image)
        # print(f"Image shape: {img_tensor.shape}")
        # Generate text input based on the current step in the plan
        text_input = f"{self.plan[self.step-1]['action']} --- {self.plan[self.step-1]['target']}"
        print(f"Text input: {text_input}")

        # run model inference
        with torch.no_grad():
            preds = self.model({"image": img_tensor.unsqueeze(0), "text": [text_input]})
            # print(f"Predictions: {preds}")
            
            # Handle stop signal if present
            if 'stop' not in preds:
                preds['stop'] = torch.tensor([-1.0], device=self.device)
            if 'step_remain' not in preds:
                preds['step_remain'] = torch.tensor([1.0], device=self.device)
            
            # Process stop signal
            stop_val = torch.sigmoid(preds['stop'])[0].cpu().item()
            step_remain = preds['step_remain'][0].cpu().item()
            
            # Extract velocity commands
            vel_cmd = torch.stack([
                preds["vx"], 
                preds["vy"], 
                preds["vz"], 
                preds["yaw"]
            ], dim=1).cpu().detach().numpy()[0]
            
            # Save image with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_filename = self.save_image(timestamp, np_image)
            # print(f"Image saved as: {img_filename}")
            # Log data to CSV
            self.log_data_to_csv(timestamp, vel_cmd, stop_val, step_remain, img_filename)
            # print(f"Data logged to CSV: {self.csv_path}")
            
            # Return velocity commands with stop information
            return vel_cmd, stop_val, step_remain
    
    def log_data_to_csv(self, timestamp, vel_cmd, stop_val, step_remain, img_filename):
        """Log flight data to CSV"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp, 
                self.step, 
                self.plan[self.step-1]['action'], 
                self.plan[self.step-1]['target'],
                vel_cmd[0], vel_cmd[1], vel_cmd[2], vel_cmd[3],
                stop_val,
                step_remain,
                img_filename
            ])

    def send_data_to_server(self, client_socket, data):
        try:
            # Create a combined array with timestamp + data
            combined_data = np.array([self.frame_count, *data], dtype=np.float32)

            print(f"Sending data to server: {combined_data}")
            
            # Convert data to bytes
            data_bytes = combined_data.tobytes()

            # Send data to the server
            client_socket.send(data_bytes)

        except:
            pass


    def reset_state(self):
        # Reset the state hidden state of the model
        if hasattr(self.model.net.stop_flagger, 'hidden_state'):
            self.model.net.stop_flagger.hidden_state = None
        # Reset the stop queue
        self.stop_queue.queue.clear()
        # Reset the frame count
        self.frame_count = 0

    def stepper(self, stop_cmd):
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            et = time.time()
            self.fps = 10 / (et - self.st) if (et - self.st) > 0 else 0
            # reset the timer and frame count
            self.st = et

    def handle_turn(self, frame):
        """Handle the Turn action separately from normal inference.
        
        Args:
            frame: The current camera frame
            
        Returns:
            tuple: (vel_cmd, stop_cmd, steps_remaining)
        """
        global current_step
        # Extract direction and angle from the target
        direction = self.plan[self.step-1]['target'].split(' ')[0]
        angle = int(self.plan[self.step-1]['target'].split(' ')[-1][:-1])
        print(f"Handling turn: {direction} {angle}°")
        
        # Calculate turn duration based on angle and rotation speed
        turn_duration = angle / (self.yr * 180 / np.pi)  # Convert rad/s to deg/s
        
        # Initialize turn start time if not already set
        if not hasattr(self, 'turn_start_time'):
            self.turn_start_time = time.time()
            # Save initial frame with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_filename = self.save_image(timestamp, frame)
            print(f"Starting {direction} turn of {angle}° for {turn_duration:.2f} seconds")
            # Log the turn start
            with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
                f.write(f"{datetime.datetime.now().isoformat()} - Starting {direction} turn of {angle}° (duration: {turn_duration:.2f}s)\n")
        
        # Check if turn is complete
        elapsed_time = time.time() - self.turn_start_time
        
        # Set velocity command based on direction
        if elapsed_time >= turn_duration:
            # Turn complete
            print(f"Turn complete after {elapsed_time:.2f} seconds")
            # Log the turn completion
            with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
                f.write(f"{datetime.datetime.now().isoformat()} - Turn complete after {elapsed_time:.2f} seconds\n")
                
            # Clean up turn state
            delattr(self, 'turn_start_time')
            vel_cmd = np.array([0, 0, 0, 0])  # Stop turning
            
            # Save final frame with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_filename = self.save_image(timestamp, frame)
            
            # Log the turn data
            self.log_data_to_csv(timestamp, vel_cmd, 1.0, 0.0, img_filename)
            
            # Move to next step
            self.step += 1
            current_step += 1
            if self.step > len(self.plan):
                print("Plan completed successfully.")
                self.complete = True
                # Log completion
                with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
                    f.write(f"{datetime.datetime.now().isoformat()} - Plan completed successfully\n")
            else:
                # Reset state and prepare for next step
                self.reset_state()
                print(f"Moving to step {self.step}: {self.plan[self.step-1]['action']}, {self.plan[self.step-1]['target']}")
        else:
            # Continue turning
            if direction == 'Left':
                vel_cmd = np.array([0, 0, 0, self.yr])
            elif direction == 'Right':
                vel_cmd = np.array([0, 0, 0, -self.yr])
            else:
                # Fallback for invalid direction
                vel_cmd = np.array([0, 0, 0, 0])
                print(f"Warning: Unknown turn direction '{direction}'")
            
            # Log periodic updates (e.g., every second)
            if not hasattr(self, 'last_turn_log') or time.time() - self.last_turn_log > 1.0:
                self.last_turn_log = time.time()
                # Save periodic frame
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filename = self.save_image(timestamp, frame)
                progress = (elapsed_time / turn_duration) * 100
                print(f"Turn progress: {progress:.1f}% ({elapsed_time:.1f}/{turn_duration:.1f}s)")
                # Log the turn data periodically
                self.log_data_to_csv(timestamp, vel_cmd, 0.0, 1.0-(elapsed_time/turn_duration), img_filename)
            time.sleep(0.1)
            
        
        # During a turn, we don't use the stop signal from the model
        stop_cmd = 0.0
        steps_remaining = 1.0
        
        return vel_cmd, stop_cmd, steps_remaining

    def run(self):
        global current_frame, trigger_new_instruction, stop_capture, current_step
        self.st = time.time() 
        while self.running and not self.complete and not stop_capture:
            if self.step != current_step:
                self.step = current_step
                self.reset_state()

            if trigger_new_instruction:
                print("New instruction triggered!")
                self.plan_n_check()
                trigger_new_instruction = False
                self.st = time.time()
                continue


            frame = current_frame
            if frame is None:   
                print("No frame captured yet. Waiting for camera to initialize...")
                time.sleep(0.1)
                continue
            
            if self.plan[self.step-1]['action'] == 'Turn':
                vel_cmd, stop_cmd, steps_remaining = self.handle_turn(frame)

            else:
                vel_cmd, stop_cmd, steps_remaining = self.inference(frame)
                vel_cmd[:-1] = vel_cmd[:-1]*4


                act = self.plan[self.step-1]['action']
                if (act == 'Right' or act == 'Left'):
                    ys = 1
                elif (act == 'Above' or act == 'Below'):
                    ys=2
                else:
                    ys = 4
                vel_cmd[-1] = vel_cmd[-1] * ys
                self.stepper(stop_cmd)

            self.send_data_to_server(self.client_socket, vel_cmd)

            if self.complete:
                break

        self.client_socket.close()
        
        # Log session end
        with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - Session ended\n")

    def start(self):
        # Log session start
        with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - Session started\n")
            
        # Start the processing thread
        print("Starting MiniNav processing thread...")
        processing_thread = threading.Thread(target=self.run, daemon=True)
        processing_thread.start()

        return processing_thread

    def stop(self):
        print("Stopping MiniNav...")
        self.running = False
        
        # Close socket properly with try-except
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
                print("Socket connection closed")
            except Exception as e:
                print(f"Error closing socket: {e}")
        
        # Write final log entry
        with open(os.path.join(self.log_dir, "events.txt"), 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - MiniNav stopped\n")


def mininav_thread_function(model_cfg_path, server_address, session_dirs):
    """Thread function to run MiniNav"""
    global stop_capture, current_frame, current_step
    
    navigator = None
    try:
        # Initialize MiniNav and start processing
        navigator = MiniNav(server_address, model_cfg_path, session_dirs)
        redy = input("Ready? Press any key")
        processing_thread = navigator.start()
        
        # Wait for main thread to signal stop
        while not stop_capture:
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in MiniNav thread: {e}")
    finally:
        # Ensure navigator is stopped
        if navigator:
            navigator.stop()
            
    print("MiniNav thread exited cleanly")
    return navigator


def main():    
    parser = argparse.ArgumentParser(description='Single Camera Stream with MiniNav Processing')
    parser.add_argument('--camera', default=0, type=int, help='Camera device number')
    parser.add_argument("--output_dir", type=str, default="mininav_sessions", help="Root directory for output files")
    parser.add_argument("--record", action='store_true', help="Save frames to output directory")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (target)")
    parser.add_argument(
        "--camera_retries",
        type=int,
        default=15,
        help="Maximum consecutive camera reopen attempts before failing.",
    )
    parser.add_argument(
        "--camera_init_timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for the first camera frame.",
    )
    
    args = parser.parse_args()

    global stop_capture, current_frame, current_step
    
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_root = os.path.join(args.output_dir, session_timestamp)
    os.makedirs(session_root, exist_ok=True)
    
    camera_frames_dir = os.path.join(session_root, "camera_frames")
    mininav_data_dir = os.path.join(session_root, "mininav_data")
    mininav_images_dir = os.path.join(mininav_data_dir, "images")
    
    if args.record:
        os.makedirs(camera_frames_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"Session started at: {session_timestamp}")
    print(f"Session root directory: {session_root}")
    print(f"Camera: {args.camera}")
    if args.record:
        print(f"Camera frames directory: {camera_frames_dir}")
        print("Recording: Enabled")
    else:
        print("Recording: Disabled")
    print("=" * 50)
    
    camera_thread = threading.Thread(
        target=camera_thread_function, 
        args=(
            args.camera,
            "current_frame",
            args.fps,
            args.camera_retries,
            args.camera_init_timeout,
        ),
        daemon=True
    )
    
    camera_thread.start()
    
    print("Waiting for camera to initialize...")
    time.sleep(min(3.0, args.camera_init_timeout))

    if stop_capture or current_frame is None:
        print(
            f"Camera initialization failed for index {args.camera}. "
            "Check /dev/video* availability or pass --camera with a valid index."
        )
        return
    
    window_name = "Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,2* 640, 2*640)
    
    print("\nPress SPACE to save current frame")
    print("Press ESC to exit")
    
    model_cfg_path = ['2025-04-29', '01-08-37', '325000']
    original_server_address = ('192.168.0.101', 8080)
    server_address = resolve_server_address(
        original_server_address[0],
        original_server_address[1],
    )

    mininav_options = {
        "session_root": session_root,
        "mininav_data_dir": mininav_data_dir,
        "images_dir": mininav_images_dir
    }

    navigator_thread = threading.Thread(
        target=mininav_thread_function, 
        args=(model_cfg_path, server_address, mininav_options),
        daemon=True
    )
    navigator_thread.start()
    
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    with open(os.path.join(session_root, "session_log.txt"), 'w') as f:
        f.write(f"Session started: {session_timestamp}\n")
        f.write(f"Camera: {args.camera}\n")
        f.write(f"Recording enabled: {args.record}\n")
        f.write(f"Model path: {model_cfg_path}\n")
        f.write(f"Server address: {server_address}\n")
    
    try:
        while not stop_capture:
            if current_frame is None:
                time.sleep(0.1)
                continue
                
            frame = current_frame.copy()
            
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                
            display_frame = frame.copy()
            cv2.putText(
                display_frame, 
                f"FPS: {fps:.2f}", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            display_frame = cv2.resize(display_frame, (2*640, 2*640))
            cv2.imshow(window_name, display_frame)
            
            if args.record:
                if frame_count % 30 == 0:
                    timestamp_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_path = os.path.join(camera_frames_dir, f"frame_{timestamp_ms}.jpg")
                    cv2.imwrite(frame_path, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                print("Exiting...")
                break
                
            if key == 32:
                print("New instruction requested!")
                global trigger_new_instruction
                trigger_new_instruction = True

            if key == 81:
                if navigator_thread.is_alive():
                    current_step = max(1, current_step - 1)
                    print(f"Step changed to: {current_step}")
            elif key == 83:
                if navigator_thread.is_alive():
                    current_step += 1
                    print(f"Step changed to: {current_step}")
                
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("Shutting down gracefully...")
        stop_capture = True
        
        with open(os.path.join(session_root, "session_log.txt"), 'a') as f:
            f.write(f"\nSession ended: {datetime.datetime.now().isoformat()}\n")
        
        print("Waiting for threads to finish...")
        navigator_thread.join(timeout=5)
        camera_thread.join(timeout=5)
        
        cv2.destroyAllWindows()
        print("Script exited cleanly")

if __name__ == "__main__":
    main()