import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import torch
import pathlib
from yolov5.detect import run


def main():

    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath

    parser = argparse.ArgumentParser(description='Predict bounding boxes on the image.')
    parser.add_argument('input_path', type=str, help='Path to input file or folder.')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to output files.')
    parser.add_argument('--model_weights', type=str, default='./yolov5/runs/train/drone_human_det4/weights/best.pt', help='Path to model weights.')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold.')
    parser.add_argument('--save_txt', action='store_true', help='Save detection results to *.txt files')

    args = parser.parse_args()

    run(
        weights=args.model_weights,
        source=args.input_path,
        project=args.output_path,
        name='',
        exist_ok=True,
        conf_thres=args.confidence,
        imgsz=(640, 640),
        save_txt=args.save_txt,
        save_conf=True,
        save_crop=False,
        nosave=False
    )



if __name__ == "__main__":
    main()