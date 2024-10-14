#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Function to get the size and position of the OpenCV window
def get_opencv_window_info(window_name="Image"):
    # Get the window position and size in pixels
    window_info = cv2.getWindowImageRect(window_name)
    x, y, width, height = window_info
    return x, y, width, height

# Mouse callback function to capture points or adjust them
def select_point(event, x, y, flags, param):
    global points, image, clone, dragging_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point selected: {x}, {y}")
            redraw_points_and_lines()
        else:
            dragging_point = find_nearest_point(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            points[dragging_point] = [x, y]
            redraw_points_and_lines()
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

def find_nearest_point(x, y):
    min_dist = float('inf')
    min_index = None
    for i, point in enumerate(points):
        dist = np.linalg.norm(np.array(point) - np.array([x, y]))
        if dist < min_dist and dist < 10:
            min_dist = dist
            min_index = i
    return min_index

def reorder_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def redraw_points_and_lines():
    global image, clone
    image = clone.copy()
    if len(points) > 0:
        for i, point in enumerate(points):
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(image, tuple(points[i - 1]), tuple(point), (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(image, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
    cv2.imshow("Image", image)

def load_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    root.destroy()
    if image_path:
        return cv2.imread(image_path)
    else:
        return None

def straighten_and_crop(image, pts):
    pts = np.array(pts, dtype="float32")
    rect = reorder_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped

def save_file(image_to_save):
    global warped  # Access the global variable to check if the warped image exists
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    root.destroy()
    if file_path:
        cv2.imwrite(file_path, image_to_save)
        print(f"Image saved at: {file_path}")
        # Close the "Warped Image" window if it exists
        if warped is not None:
            cv2.destroyWindow("Warped Image")
            cv2.destroyWindow("Image")
            
            

def rotate_image():
    global image, clone, angle
    angle += 90
    image = cv2.rotate(clone, cv2.ROTATE_90_CLOCKWISE)
    clone = image.copy()
    cv2.imshow("Image", image)

def crop_image():
    global warped
    if len(points) == 4:
        warped = straighten_and_crop(clone, points)
        cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Warped Image", warped)
    else:
        print("Please select 4 points first.")

def redo_image():
    global points, image, clone
    points = []
    image = clone.copy()
    cv2.imshow("Image", image)



def upload_new_image():
    global image, clone, points
    points = []
    image = load_image()
    if image is not None:
        clone = image.copy()
    
          # Set up the OpenCV window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image", select_point)
        cv2.imshow("Image", image)

    else:
        print("No image selected.")


def main():
    global image, clone, points

    image = load_image()

    if image is None:
        print("No image selected.")
        return

    clone = image.copy()

    # Set up the OpenCV window
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", select_point)
    cv2.imshow("Image", image)


    # Allow time for the OpenCV window to appear
    time.sleep(1)

    # Get the position and size of the OpenCV window
    x, y, width, height = get_opencv_window_info()

    # Set up the Tkinter GUI for buttons
    root = tk.Tk()
    root.title("Image Editing")

    # Position the Tkinter window near the OpenCV window without overlapping
    # We place it to the right of the OpenCV window
    root.geometry(f"200x230+{x + width + 10}+{y}")  # 10 pixels to the right of OpenCV window

    button_frame = tk.Frame(root)
    button_frame.pack(side="bottom", pady=10)

    rotate_button = Button(button_frame, text="Rotate", command=rotate_image, width=15, height=2)
    rotate_button.pack(side="bottom", padx=5)

    crop_button = Button(button_frame, text="Crop", command=crop_image, width=15, height=2)
    crop_button.pack(side="bottom", padx=5)
    
    redo_button = Button(button_frame, text="Redo", command=redo_image, width=15, height=2)
    redo_button.pack(side="bottom", padx=5)
    
    save_button = Button(button_frame, text="Save", command=lambda: save_file(warped if warped is not None else image), width=15, height=2)
    save_button.pack(side="bottom", padx=5)

    new_image_button = Button(button_frame, text="Upload New Image", command=upload_new_image, width=15, height=2)
    new_image_button.pack(side="bottom", padx=5)

    # Run the Tkinter main loop
    root.mainloop()

    # Close all OpenCV windows on exit
    cv2.destroyAllWindows()

# Run the main function
main()

