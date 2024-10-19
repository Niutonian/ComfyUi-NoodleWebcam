## ComfyUi-NoodleWebcam
Noodle webcam is a node that records frames and send them to your favourite node.

![NoodleNode](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/noodlenode.png)


Currently you can only select the webcam, set the frame rate, set the duration and start/stop the stream (for continuous streaming TODO). 

## BROKEN THINGS// ANY HELP TO FIX THESE ISSUES WOULD BE MORE THAN WELCOME
- This node is completely broken because you need to change the framerate or duration to get the node to restart.
- It will randomly crash and need a restart of comfyUI
- doesn't return the correct preview
- it crashes a lot so let me know what kind of error you are getting

## EXAMPLE NODE
![Tiger workflow](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/webcamNode.png)

## WORKFLOW
Example of output
![Tiger workflow](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/workflow.png)

OpenPoseDW Controlnet

![WebcamToOpenPoseDW](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/webcamNode_OpPDW.gif)

Depth Controlnet

![WebcamToDepth](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/webcamNode_depth.gif)

Depth Controlnet invert node on

![WebcamToDepthInvert](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/webcamNode_depth_invert.gif)

Have fun

![WebcamNode](https://github.com/Niutonian/ComfyUi-NoodleWebcam/blob/main/workflow/webcamNode.gif)


## PREREQUISITES, NEEDED CUSTOM NODES

AnimateDiff Evolved -  https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved

pythongosssss ComfyUI-Custom-Scripts -  https://github.com/pythongosssss/ComfyUI-Custom-Scripts

MTB Nodes -  https://github.com/melMass/comfy_mtb

ComfyUI FizzNodes - https://github.com/FizzleDorf/ComfyUI_FizzNodes

ComfyUI Advanced ControlNet -  https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet

ComfyUI's ControlNet Auxiliary Preprocessors - https://github.com/Fannovel16/comfyui_controlnet_aux

Latent Consistency Model for ComfyUI - https://github.com/0xbitches/ComfyUI-LCM

## INSTALLATION
To install clone this repo inside your confyUI's custom_nodes folder

## TODO
-- Fix all the bugs above

-- Continuous streaming
