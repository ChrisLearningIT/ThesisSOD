from ultralytics import YOLO  # Import the YOLO model from Ultralytics
from nn.modules.conv import SpinningConv
import torch


# # Initialize your customised yolov8tiny model
# # Make sure the path to your model.yaml is correct
yolov8tiny_model = YOLO('cfg/models/v8/Yolov8tiny.yaml')
results = yolov8tiny_model.train(data="coco8.yaml", epochs=3)
results = yolov8tiny_model.val()

results = yolov8tiny_model("https://ultralytics.com/images/bus.jpg")
