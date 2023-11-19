import cv2
import numpy as np
import torch
import time
from PIL import Image
import torchvision.transforms.functional as TF  # Importing the required function
from torchvision.transforms.functional import to_pil_image

# Hello, this code was jerryrigged together over a few days by mixing a few things I have learned online and a little chatGPT magik, 
# so don't hate me if it breaks, I'll try my best to update and fix the issues
# Have fun,with it 
# Neofuturism

class WebcamNode:
 
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                     "select_webcam": ("INT", {
#                     "default": 0, 
#                     "min": 0, # Minimum value, assuming webcam IDs start at 0
#                     "max": 10, # Arbitrary max value, can be adjusted based on the expected number of webcams
#                     "step": 1, 
#                     "display": "number"
#                 }),
#                 "framerate": ("INT", {
#                     "default": 12, 
#                     "min": 1,
#                     "max": 60, 
#                     "step": 1, 
#                     "display": "slider"
#                 }),
#                 "control_stream": (["start", "stop"],),
#                 "duration": ("INT", {
#                     "default": 10, 
#                     "min": 1,
#                     "max": 120, 
#                     "step": 1, 
#                     "display": "slider"
#                 }),
#             },
#         }


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_webcam": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
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
                "control_stream": ("STRING", {
                    "choices": ["start", "stop"],
                    "default": "start"
                }),
                "duration": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "slider"
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 100,
                    "max": 1080,
                    "step": 1,
                    "display": "number"  # Allows specifying the height of the captured frame
                }),
                "width": ("INT", {
                    "default": 640,
                    "min": 100,
                    "max": 1920,
                    "step": 1,
                    "display": "number"  # Allows specifying the width of the captured frame
                })
            },
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "capture"
    CATEGORY = "🍜Noodle Webcam"

    def __init__(self):
        self.webcam = None
        self.active = False


# THIS IS WORKING BUT THE IMAGES ARE WHITE 

    def capture(self, select_webcam, framerate, control_stream, duration, height, width):
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
                        # Resize frame to the given dimensions
                        resized_frame = cv2.resize(frame, (width, height))

                        # Convert resized frame to RGB
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite('test_frame.jpg', rgb_frame)  # Save the frame for inspection

                        # Convert to tensor and normalize
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
                    return (torch.empty(0, height, width, 3, dtype=torch.uint8),)  # Update the shape

        except Exception as e:
            print(f"Error in WebcamNode: {e}")
            if self.webcam:
                self.webcam.release()
            self.active = False
            return (torch.empty(0, height, width, 3, dtype=torch.uint8),)  # Update the shape

        return (torch.empty(0, height, width, 3, dtype=torch.uint8),)  # Update the shape and return an empty tensor if not active





# THIS ONE WORKS BUT RENDERS A 1000 FRAMES

    def captureBroken(self, select_webcam, framerate, control_stream, duration, height, width):
        frames = []
        try:
            if control_stream == "start" and not self.active:
                self.webcam = cv2.VideoCapture(select_webcam)
                self.webcam.set(cv2.CAP_PROP_FPS, framerate)
                self.active = True
                self.start_time = time.time()

            frame_count = 0
            total_frames = int(duration * framerate)
            expected_time_per_frame = 1.0 / framerate
            next_frame_time = self.start_time

            if self.active:
                while frame_count < total_frames:
                    current_time = time.time()
                    if current_time >= next_frame_time:
                        ret, frame = self.webcam.read()
                        if ret:
                            # Resize frame to the given dimensions
                            resized_frame = cv2.resize(frame, (width, height))

                            # Convert resized frame to RGB
                            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                            cv2.imwrite('test_frame.jpg', rgb_frame)  # Save the frame for inspection

                            # Convert to tensor and normalize
                            rgb_tensor = torch.from_numpy(rgb_frame).float() / 255.0
                            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # Rearrange and add batch dimension
                            frames.append(rgb_tensor)

                            frame_count += 1
                            print(f"Captured Frame: {frame_count}/{total_frames}")

                            next_frame_time = self.start_time + frame_count * expected_time_per_frame
                        else:
                            break

            self.webcam.release()
            self.active = False

            if frames:
                all_frames = torch.stack(frames).squeeze(1)  # Remove any extra dimension of size 1
                print("Tensor shape before processing:", all_frames.shape)

                # Check if the tensor has the correct shape (N, C, H, W)
                if all_frames.ndim != 4 or all_frames.shape[1] != 3:
                    # Reshape or reorder dimensions here as needed
                    # Example: all_frames = all_frames.view(-1, 3, height, width)
                    pass

                all_frames = all_frames.permute(0, 2, 3, 1)  # [N, H, W, C]
                all_frames = torch.nn.functional.interpolate(all_frames, size=(height, width), mode='bilinear', align_corners=False)
                all_frames = all_frames.permute(0, 3, 1, 2)
                print("Tensor shape after processing:", all_frames.shape)
                return all_frames

            else:
                print("No frames captured.")
                return torch.empty(0, 3, height, width, dtype=torch.float32)

        except Exception as e:
            print(f"Error in WebcamNode: {e}")
            if self.webcam:
                self.webcam.release()
            self.active = False
            return torch.empty(0, 3, height, width, dtype=torch.float32)

        return torch.empty(0, 3, height, width, dtype=torch.float32)  # Return an empty tensor if not active


# THIS IS BROKEN TOO

    def captureBROKEN_B(self, select_webcam, framerate, control_stream, duration, height, width):
        frames = []
        try:
            if control_stream == "start" and not self.active:
                self.webcam = cv2.VideoCapture(select_webcam)
                self.webcam.set(cv2.CAP_PROP_FPS, framerate)
                self.active = True
                self.start_time = time.time()

            frame_count = 0
            total_frames = int(duration * framerate)

            if self.active:
                while frame_count < total_frames:
                    ret, frame = self.webcam.read()
                    if ret:
                        # Resize frame to the given dimensions
                        resized_frame = cv2.resize(frame, (width, height))

                        # Convert resized frame to RGB
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite('test_frame.jpg', rgb_frame)  # Save the frame for inspection

                        # Convert to tensor and normalize
                        rgb_tensor = torch.from_numpy(rgb_frame).float() / 255.0
                        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # Rearrange and add batch dimension
                        frames.append(rgb_tensor)


                        frame_count += 1
                        print(f"Captured Frame: {frame_count}/{total_frames}")  # Debugging print statement
                    else:
                        break
                    time.sleep(1.0 / framerate)

                self.webcam.release()
                self.active = False

            if frames:
                all_frames = torch.stack(frames)
                print("Tensor shape before processing:", all_frames.shape)

                # Ensure the tensor is in the correct shape [N, C, H, W]
                if all_frames.ndim != 4 or all_frames.shape[1] != 3:
                    # Reshape logic here if necessary
                    all_frames = all_frames.view(-1, 3, height, width)

                # Permute the tensor if necessary for interpolation
                all_frames = all_frames.permute(0, 2, 3, 1)  # [N, H, W, C]

                # Apply interpolation
                all_frames = torch.nn.functional.interpolate(all_frames, size=(height, width), mode='bilinear', align_corners=False)

                # Permute back to [N, C, H, W]
                all_frames = all_frames.permute(0, 3, 1, 2)

                print("Tensor shape after processing:", all_frames.shape)
                return all_frames
            else:
                return all_frames if frames else torch.empty(0, 3, height, width, dtype=torch.float32)

        except Exception as e:
            print(f"Error in WebcamNode: {e}")
            if self.webcam:
                self.webcam.release()
            self.active = False
            return torch.empty(0, 3, height, width, dtype=torch.float32)

        return torch.empty(0, 3, height, width, dtype=torch.float32)  # Return an empty tensor if not active


        
# Node registration
NODE_CLASS_MAPPINGS = {
    "WebcamNode": WebcamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamNode": "🍜Webcam Noodle"
}

