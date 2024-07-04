
import cv2
import os

# Directory containing images
img_dir = r"C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\HandImages"

# Output directory for cropped images
output_dir = r"C:\Users\Melissa\AppData\Local\Programs\Python\Python312\objdetect\finalimgs"
os.makedirs(output_dir, exist_ok=True)

# Function to crop and save the image
def crop_and_save_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return

    # Display the image
    cv2.imshow("Image", image)
    # Wait for the user to select a region and press Enter
    r = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
    if r != (0, 0, 0, 0):
        cropped_image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        if cropped_image.size == 0:
            print(f"Empty cropped image for {image_path}. Skipping...")
            return
        
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to {output_path}")

    cv2.destroyAllWindows()

# Process each image in the directory and its subdirectories
for root, dirs, files in os.walk(img_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            relative_path = os.path.relpath(image_path, img_dir)
            output_path = os.path.join(output_dir, f"cropped_{relative_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            crop_and_save_image(image_path, output_path)
