#!/usr/bin/env python
# coding: utf-8

# In[8]:


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

    # Difference between x and y (top-right will have the smallest difference, bottom-left will have the largest)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

# Function to redraw the points and connecting lines on the image
def redraw_points_and_lines():
    global image, clone
    image = clone.copy()  # Reset the image

    # Draw circles for points and lines connecting them
    for i, point in enumerate(points):
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(image, tuple(points[i - 1]), tuple(points[i]), (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(image, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

    cv2.imshow("Image", image)

# Function to perform the perspective transformation and crop
def straighten_and_crop(image, pts):
    # Reorder the points to ensure the correct order
    pts = reorder_points(np.array(pts, dtype="float32"))

    # Extract the four ordered points
    (tl, tr, br, bl) = pts

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for perspective transformation
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def select_file():
    # Use tkinter file dialog to select the image file
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    return file_path

def save_file():
    # Use tkinter file dialog to select the save location
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")],
                                             title="Save Edited Image As")
    return save_path

# Main function to handle image loading, point selection, and cropping
def main():
    global image, clone, points, dragging_point, warped

    # Step 1: Let the user select an image file
    image_path = select_file()
    if not image_path:
        print("No image file selected.")
        return

    # Step 2: Read and display the image
    image = cv2.imread(image_path)
    clone = image.copy()  # Clone the image to preserve the original for processing
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", select_point)

    print("Please click on four corners in the image. Drag the points to adjust them as needed.")

    # Step 3: Wait until four points are selected and adjusted
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # Once 4 points are selected, wait for 'c' to perform cropping
        if len(points) == 4:
            print("Four points selected. Adjust the points and press 'c' to crop.")
            break

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


