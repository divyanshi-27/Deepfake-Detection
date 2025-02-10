import cv2  
import sys  

# Check if image name is provided  
if len(sys.argv) < 2:  
    print("Usage: python image_processing.py <image_name>")  
    sys.exit(1)  

image_name = sys.argv[1]  # Get the image name from command line  

# Load the image  
img = cv2.imread(image_name)  

if img is None:  
    print(f"Error: Cannot load {image_name}. Check the file name.")  
else:  
    print(f"Loaded: {image_name}")  
    cv2.imshow("Loaded Image", img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
