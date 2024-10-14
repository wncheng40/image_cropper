#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Initialize a list to store points and track the currently selected point
points = []
dragging_point = None
image = None
clone = None

# Mouse callback function to capture points or adjust them
def select_point(event, x, y, flags, param):
    global points, image, clone, dragging_point

    # On left mouse button click, if fewer than 4 points, capture the point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point selected: {x}, {y}")

            # Draw the circle and lines on the image
            redraw_points_and_lines()

        else:
            # Check if an existing point is clicked to start dragging
            dragging_point = find_nearest_point(x, y)

    # On mouse move, adjust the dragging point's location
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            points[dragging_point] = [x, y]
            redraw_points_and_lines()

    # On mouse release, finalize the adjustment
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

# Function to find the nearest point to the current mouse click for dragging
def find_nearest_point(x, y):
    min_dist = float('inf')
    min_index = None
    for i, point in enumerate(points):
        dist = np.linalg.norm(np.array(point) - np.array([x, y]))
        if dist < min_dist and dist < 10:  # If click is near a point
            min_dist = dist
            min_index = i
    return min_index

# Function to reorder points to ensure top-left, top-right, bottom-right, bottom-left order
def reorder_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of x + y (top-left will have the smallest sum, bottom-right will have the largest sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference between x and y (top-right will have the smallest difference, bottom-left the largest)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

# Function to redraw points and lines on the image
def redraw_points_and_lines():
    global image, clone
    image = clone.copy()

    # Draw the points and lines connecting them
    if len(points) > 0:
        for i, point in enumerate(points):
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)  # Draw circle
            if i > 0:
                cv2.line(image, tuple(points[i - 1]), tuple(point), (0, 255, 0), 2)  # Draw line
        if len(points) == 4:
            cv2.line(image, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)  # Close the quadrilateral

    cv2.imshow("Image", image)

# Function to load an image using a file dialog
def load_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename()
    root.destroy()

    if image_path:
        return cv2.imread(image_path)
    else:
        return None

# Function to straighten and crop the selected area from the image
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

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped

# Function to save the file using a file dialog
def save_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    root.destroy()
    return file_path

# Main function to run the image cropping program
def main():
    global image, clone, points

    # Step 1: Load the image
    image = load_image()

    if image is None:
        print("No image selected.")
        return

    clone = image.copy()

    # Step 2: Set up the window and mouse callback
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Make the window resizable
    cv2.setMouseCallback("Image", select_point)

    cv2.imshow("Image", image)

    # Step 3: Let the user select 4 points
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the points
        if key == ord("r"):
            image = clone.copy()
            points = []
            print("Reset points.")
            cv2.imshow("Image", image)

        # Press 'q' to quit without processing
        if key == ord("q"):
            cv2.destroyAllWindows()
            return

    # Wait for the user to press 'c' to crop
    while True:
        key = cv2.waitKey(0) & 0xFF

        # Press 'c' to crop the image using the adjusted points
        if key == ord("c"):
            warped = straighten_and_crop(clone, points)
            cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)  # Make the cropped window resizable
            cv2.imshow("Warped Image", warped)
            break

        # Press 'r' to redo points
        elif key == ord("r"):
            image = clone.copy()
            points = []
            print("Reset points.")
            cv2.imshow("Image", image)
            break

        # Press 'q' to quit without processing
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return

    # Step 4: Allow the user to save or redo the points after cropping
    while True:
        print("Press 's' to save, 'r' to redo points, or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF

        # If user presses 's', let them save the image
        if key == ord("s"):
            save_path = save_file()
            if save_path:
                cv2.imwrite(save_path, warped)
                print(f"Straightened image saved at: {save_path}")
            break

        # If user presses 'r', reset the points and restart the selection
        elif key == ord("r"):
            image = clone.copy()
            points = []
            cv2.imshow("Image", image)
            print("Redo points selection.")
            break

        # If user presses 'q', quit without saving
        elif key == ord("q"):
            print("Quit without saving.")
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Run the main function
main()

