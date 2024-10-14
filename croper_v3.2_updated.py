
#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Button
import time

# Initialize global variables
points = []
dragging_point = None
image = None
clone = None
warped = None
angle = 0
scale_factor = 1.0  # Initial scale factor for zooming
min_scale = 0.2     # Minimum zoom out factor
max_scale = 5.0     # Maximum zoom in factor

# Function to get the size and position of the OpenCV window
def get_opencv_window_info(window_name="Image"):
    window_info = cv2.getWindowImageRect(window_name)
    x, y, width, height = window_info
    return x, y, width, height

# Mouse callback function to capture points, adjust them, or zoom
def select_point(event, x, y, flags, param):
    global points, image, clone, dragging_point, scale_factor, min_scale, max_scale

    # Mouse left button down: Select or move points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point selected: {x}, {y}")
            redraw_points_and_lines()
        else:
            dragging_point = find_nearest_point(x, y)

    # Mouse move: Dragging selected point
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            points[dragging_point] = [x, y]
            redraw_points_and_lines()

    # Mouse left button up: Stop dragging
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

    # Mouse wheel event: Zoom in or out
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll up, zoom in
            scale_factor = min(scale_factor * 1.1, max_scale)
        else:  # Scroll down, zoom out
            scale_factor = max(scale_factor * 0.9, min_scale)

        # Resize the image according to the scale factor
        height, width = clone.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        image = cv2.resize(clone, new_size, interpolation=cv2.INTER_LINEAR)
        redraw_points_and_lines()

# Function to find the nearest point within a certain radius for dragging
def find_nearest_point(x, y):
    min_dist = float('inf')
    min_index = None
    for i, point in enumerate(points):
        dist = np.linalg.norm(np.array(point) - np.array([x, y]))
        if dist < min_dist and dist < 10:  # 10-pixel threshold
            min_dist = dist
            min_index = i
    return min_index

# Function to redraw the points and lines on the image
def redraw_points_and_lines():
    global image, clone
    image = clone.copy()
    if len(points) > 0:
        for i, point in enumerate(points):
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(image, tuple(points[i-1]), tuple(point), (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(image, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

# Main function to run the application
def main():
    global image, clone

    # Load an image using file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    image_path = filedialog.askopenfilename(title='Select an Image')
    if not image_path:
        print("No image selected.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load the image.")
        return

    clone = image.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", select_point)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
