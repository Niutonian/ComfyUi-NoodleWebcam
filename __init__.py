import cv2
import numpy as np
import torch
import time
from PIL import Image

# Hello, this code was jerryrigged together over a few days by mixing a few things I have learned online and a little chatGPT magik, 
# so don't hate me if it breaks, I'll try my best to update and fix the issues
# Have fun,with it 
# Neofuturism

class WebcamNode:
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "select_webcam": ("INT", {
                    "default": 0, 
                    "min": 0, # Minimum value, assuming webcam IDs start at 0
                    "max": 10, # Arbitrary max value, can be adjusted based on the expected number of webcams
                    "step": 1, 
                    "display": "number"
                }),
                "framerate": ("INT", {
                    "default": 12, 
                    "min": 1,
                    "max": 60, 
                    "step": 1, 
                    "display": "slider"
                }),
                "control_stream": (["start", "stop"],),
                "duration": ("INT", {
                    "default": 10, 
                    "min": 1,
                    "max": 120, 
                    "step": 1, 
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "capture"
    CATEGORY = "🍜Noodle Webcam"

    def __init__(self):
        self.webcam = None
        self.active = False


    def capture(self, select_webcam, framerate, control_stream, duration):
        frames = []
        try:
            if control_stream == "start" and not self.active:
                self.webcam = cv2.VideoCapture(select_webcam)
                self.webcam.set(cv2.CAP_PROP_FPS, framerate)
                self.active = True
                self.start_time = time.time()

            if self.active:
                while time.time() - self.start_time <= duration:
                    ret, frame = self.webcam.read()
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite('test_frame.jpg', rgb_frame)  # Save the frame for inspection
                        rgb_tensor = torch.from_numpy(rgb_frame).float() / 255.0
                        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # Rearrange and add batch dimension
                        frames.append(rgb_tensor)
                    else:
                        break
                    time.sleep(1.0 / framerate)

                self.webcam.release()
                self.active = False

                if frames:
                    # Concatenate and reshape tensor to (N, H, W, C)
                    all_frames = torch.cat(frames, dim=0)
                    all_frames = all_frames.permute(0, 2, 3, 1)  # Reshape to (N, H, W, C)
                    all_frames = all_frames.mul(255).byte()  # Scale to 0-255 and convert to byte
                    return (all_frames,)
                else:
                    return (torch.empty(0, 768, 512, 3, dtype=torch.uint8),)

        except Exception as e:
            print(f"Error in WebcamNode: {e}")
            if self.webcam:
                self.webcam.release()
            self.active = False
            return (torch.empty(0, 768, 512, 3, dtype=torch.uint8),)

        return (torch.empty(0, 768, 512, 3, dtype=torch.uint8),)  # Return an empty tensor if not active

   

# Node registration
NODE_CLASS_MAPPINGS = {
    "WebcamNode": WebcamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamNode": "🍜Webcam Noodle"
}
