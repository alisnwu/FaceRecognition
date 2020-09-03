import cv2
import os #for reading training data directories and paths
import numpy as np #to convert python lists to numpy arrays as it is needed by OpenCV face recognizers

subjects = ["", "Alison"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the test image to gray image as opencv face detector expects gray images
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5); #to detect images in different scales
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0] #extract the face area
    return gray[y:y+w, x:x+h], faces[0] #return only the face part of the image

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this will read all the training images
#and detech the face from each image
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
                draw_rectangle(image, rect)
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(200)
            else:
                print("No face found in "+ image_path)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()  
    return faces, labels

#there are 2 lists
#one contains all the faces
#and the other contains respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#create face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#train our face recognizer
face_recognizer.train(faces, np.array(labels))

#this recognizes the person in image passed
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
    return img

print("Predicting images...")

#perform a prediction
test_img1 = cv2.imread("test-data/test1.jpg")
predicted_img1 = predict(test_img1)
print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (800, 500)))

#predict video from webcam
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    predicted = predict(img)
    cv2.imshow('Face Recognition', predicted)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
