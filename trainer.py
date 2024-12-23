import os
import cv2
import numpy as np
from PIL import Image

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset
path = 'dataSet'

def getImagesWithID(path):
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist!")
        exit(1)

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        try:
            faceImg = Image.open(imagePath).convert('L')  # Convert to grayscale
            faceNp = np.array(faceImg, 'uint8')  # Convert to NumPy array
            ID = int(os.path.split(imagePath)[-1].split('.')[1])  # Extract ID
            faces.append(faceNp)
            IDs.append(ID)
            print(f"Training with ID: {ID}")
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        except (IndexError, ValueError):
            print(f"Skipping file with incorrect format: {imagePath}")
            continue
    return np.array(IDs), faces

# Collect IDs and faces, then train
ids, faces = getImagesWithID(path)
recognizer.train(faces, ids)

# Save trained recognizer
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainingData.yml')

# Cleanup
cv2.destroyAllWindows()