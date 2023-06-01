from imutils import paths
import face_recognition
# from face_recognition import face_locations, face_encodings
import pickle
import cv2 as cv

# Add your names, as the same as the name of your folders
current_names = ['parentsacc','kid', 'parent']
predefined_encodings = []
predefined_names = []

for current_name in current_names:
    print(f"Start training process for {current_name}...")

    # getting all images paths for this person
    image_paths = list(paths.list_images(f"CollectedFaces/{current_name}"))

    # loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        print(f"Processing image {i+1} for {current_name}")

        # load and convert image from BGR to RGB
        image = cv.imread(image_path)
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            predefined_encodings.append(encoding)
            predefined_names.append(current_name)

data = {"encodings": predefined_encodings, "names": predefined_names}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))