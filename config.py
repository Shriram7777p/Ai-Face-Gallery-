#import file part
import os

# Configuration settings
INPUT_FOLDER = 'test_image' #input folder 
OUTPUT_FOLDER = 'output_image' #output save folder
SRC_FOLDER = 'source_image' #face want to comparision folder
SIMILARITY_THRESHOLD = 0.6 

# Get all image files from the SRC_FOLDER
SRC_IMAGE_PATHS = [os.path.join(SRC_FOLDER, f) for f in os.listdir(SRC_FOLDER) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]