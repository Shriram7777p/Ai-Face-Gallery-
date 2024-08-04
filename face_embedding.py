#import file part
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from face_detection import align_face
from retinaface import RetinaFace

# Set fixed random seed for reproducibility
torch.manual_seed(42)

#to set the device use GPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#embedding the face of input_folder
def extract_embedding(face_img, landmarks):
    try:
        aligned_face = align_face(face_img, landmarks)
        face_img = cv2.resize(aligned_face, (160, 160))
        face_img = transforms.ToTensor()(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_img).cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        print(f"Error in extract_embedding: {str(e)}")
        return None

#extraction 
def extract_face_embedding(image_path, use_multiple_references=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(image_rgb)
    
    if not faces:
        raise ValueError(f"No face detected in the image: {image_path}")
    
    embeddings = []
    for face in faces.values():
        (x1, y1, x2, y2) = face['facial_area']
        face_img = image_rgb[y1:y2, x1:x2]
        landmarks = {
            'left_eye': face['landmarks']['left_eye'],
            'right_eye': face['landmarks']['right_eye']
        }
        embedding = extract_embedding(face_img, landmarks)
        
        if embedding is not None:
            embeddings.append(embedding)
    
    if not embeddings:
        raise ValueError(f"Failed to extract embeddings from faces in: {image_path}")
    
    if use_multiple_references:
        return embeddings
    else:
        return embeddings[0]