# File: test_display.py
import cv2
# --- Make sure this filename is PERFECTLY correct ---
filename = './img/camera_man.jpg'
# ----------------------------------------------------
print(f"Attempting to load '{filename}'...")
# Load the image
image = cv2.imread(filename)
# Check if the image was loaded successfully
if image is None:
    print(f"FAILED to load the image.")
    print("Checklist:")
    print("1. Is the file really named '" + filename + "'?")
    print("2. Is it in the same folder as this script?")
    print("3. Is the file a valid, uncorrupted image?")
else:
    print("Image loaded successfully!")
    # Display the image in a window
    cv2.imshow("Test Image", image)
    
    print("A window named 'Test Image' should appear. Press any key to close it.")
    
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Window closed.")