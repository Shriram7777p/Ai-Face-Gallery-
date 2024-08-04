#import file path
import os
import cv2
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from face_embedding import extract_face_embedding
from retinaface import RetinaFace

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

#AI process Proecess part
def process_folder(input_folder, output_folder, src_image_paths, similarity_threshold=0.8):
    processed_images = set()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    unique_images = {}
    duplicates_removed = 0

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        image_hash = get_image_hash(image_path)
        if image_hash not in unique_images:
            unique_images[image_hash] = filename
        else:
            os.remove(image_path)
            duplicates_removed += 1
            print(f"Removed duplicate: {filename}")

    print(f"Removed {duplicates_removed} duplicate images")

    try:
        src_embeddings = [extract_face_embedding(path, use_multiple_references=True) for path in src_image_paths]
        src_embeddings = [emb for sublist in src_embeddings for emb in sublist]
        print(f"Reference face embeddings extracted successfully: {len(src_embeddings)} embeddings")
    except Exception as e:
        print(f"Error extracting reference face embeddings: {str(e)}")
        return

    total_files = len(unique_images)
    matches_found = 0

    for index, (_, filename) in enumerate(unique_images.items(), 1):
        print(f"Processing image {index}/{total_files}: {filename}")

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {filename}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = RetinaFace.detect_faces(image_rgb)
        image_has_match =False
        
        for face_key, face_data in faces.items():
            if image_has_match:
                break 
            
            (x1, y1, x2, y2) = face_data['facial_area']
            face_img = image_rgb[y1:y2, x1:x2]

            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue

            try:
                landmarks = {
                    'left_eye': face_data['landmarks']['left_eye'],
                    'right_eye': face_data['landmarks']['right_eye']
                }
                embedding = extract_face_embedding(image_path)
                if embedding is None:
                    continue

                similarities = [cosine_similarity(src_emb.reshape(1, -1), embedding.reshape(1, -1))[0][0] for src_emb in src_embeddings]
                max_similarity = max(similarities)
                
                if max_similarity > similarity_threshold:
                    if filename not in processed_images:
                        matches_found += 1
                        print(f"Match found in {filename}. Similarity: {max_similarity:.4f}. Total matches: {matches_found}")
                        output_path = os.path.join(output_folder, f"match_{matches_found}_{filename}")
                        cv2.imwrite(output_path, image)  # Save in normal color format
                        processed_images.add(filename)
                        image_has_match = True
                    else:
                        print(f"Additional match found in already processed image: {filename}")
            except Exception as e:
                print(f"Error processing face in {filename}: {str(e)}")

    print(f"Processing complete. Total matching images found: {matches_found}")