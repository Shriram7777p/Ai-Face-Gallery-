#import file part
from config import INPUT_FOLDER, OUTPUT_FOLDER, SRC_IMAGE_PATHS, SIMILARITY_THRESHOLD
from image_processing import process_folder

#start Process
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, SRC_IMAGE_PATHS, SIMILARITY_THRESHOLD)