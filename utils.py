import numpy as np
import cv2
import glob
import time
import dlib
import configparser

from imutils import face_utils
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

config = configparser.ConfigParser()
config.read('example.ini')

import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

def config_video_save(img, fps, name):
    height, width, layers = img.shape
    size = (width, height)

    return cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

def load_images_from_database(path, format_img):
    # Getting names, images and feature vector from database
    folders = glob.glob(path)
    image_names_list = []
    vector_names_list = []
    for folder in folders:
        for f in glob.glob(folder + format_img):
            image_names_list.append(f)

        for f2 in glob.glob(folder + '/*.npy'):
            vector_names_list.append(f2)

    read_images = []
    read_vector = []
    for image in image_names_list:
        read_images.append(cv2.imread(image))

    for vector in vector_names_list:
        read_vector.append(np.load(vector))

    return read_images, image_names_list, read_vector, vector_names_list

def initialization_exp_filter(img_array, bgFilter = 'MEAN'):
    # We get the shape of the images
    size = img_array[0].shape[0:2]

    # Initial background calculation
    background = np.zeros(shape = size)
    h, w = size
    temp = np.zeros(shape = (len(img_array), h, w))
    for i in range(len(img_array)):
        frame = img_array[i]
        temp[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (bgFilter == 'MEAN'):
        background = temp.mean(axis = 0).astype(np.uint8)
    elif (bgFilter == 'MEDIAN'):
        background = np.median(temp, axis = 0).astype(np.uint8)

    return background

def exponentialFilter(frame, alpha, morph = True, morph_kernel = (4, 4), background_init = 0):
    """
    Calculates the background subtraction using the exponential filter approach.
    :param alpha: learning rate [0,1]. Value = 0 means background is not update, value = 1 means the new frame is set as
    background
    :param morph: True to apply morphological operation to the resulting background subtraction
    :param morph_kernel: kernel to be applied with the morphological operation
    :return: foreground and background
    """

    # Initialize kernel for morphological transformation (opening)
    if (morph):
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)

    # Take the next frame/image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract the background from the frame to get the foreground (out)
    out = np.abs(frame - background_init)
    ret, out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
    if (morph):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, strel).astype(np.uint8)

    # Calculate the new background
    background = ((1 - alpha) * background_init + alpha * frame).astype(np.uint8)

    return out, background

def load_face_detector(num):
    if num == 1:    # HOG + SVM detector
        return dlib.get_frontal_face_detector()
    elif num == 2:  # CNN detector
        return dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
    else:   # Viola-Jones detector
        return cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

def face_detection(num, face_detector, img, coords, bs):
    x, y, w, h = coords
    if num == 3:    # For Viola-Jones detector
        if bs:
            return face_detector.detectMultiScale(img[y:y+h, x:x+w])
        else:
            return face_detector.detectMultiScale(img)

    else: # For HOG + SVM and CNN detector
        if bs:
            return face_detector(img[y:y+h, x:x+w], 1)
        else:
            return face_detector(img, 1)

def getROI(image, th):
    # Find the max area contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rois = []
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < th:
            continue
        rois.append(contours[c])

    return image_color, rois

def draw_ROI(roi, img):
    # Draw ROI and returning its coordinates in format x, y, w, h.
    x, y, w, h = cv2.boundingRect(roi)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, (np.int32(x + (w // 2)), np.int32(y + (h // 2))), 2, (255, 0, 0), -1)
    coordinates = (x, y, w, h)

    return img, coordinates

def correction_face_detection(faces, img, coords, num, bs):
    # Depending algorithm face detection used, we need to apply conversions
    x, y, w, h = coords
    good_face = []
    coords_face = []
    rect_face = []
    bad_faces = 0

    for faceRect in faces:
        if num == 2:
            faceRect = faceRect.rect
        if num == 3:
            (column, row, width, height) = faceRect
            faceRect = dlib.rectangle(column, row, width + column, height + row)

        x_ = faceRect.left()
        y_ = faceRect.top()
        w_ = faceRect.right() - x_
        h_ = faceRect.bottom() - y_

        if bs:
            bbox = img[y:y + h, x:x + w]
            cara = bbox[y_:y_ + h_, x_:x_ + w_].copy()
        else:
            cara = img[y_:y_ + h_, x_:x_ + w_].copy()

        # Sometimes, its detected faces with shape = 0 in some dimension
        if cara.shape[0] != 0 and cara.shape[1] != 0:
            good_face.append(cara)
            coords_face.append((x_, y_, w_, h_))
            rect_face.append(faceRect)
        else:
            bad_faces += 1

    return good_face, coords_face, rect_face

def draw_faces_and_landmarks(coords, coords_faces, rect_faces, img, predictor, bs):
    # For each face detected, draw a rectangle and its landmarks
    x, y, w, h = coords
    for i in range(len(coords_faces)):

        x_, y_, w_, h_ = coords_faces[i]
        if bs:
            # Draw a rectangle
            cv2.rectangle(img[y:y+h, x:x+w], (x_, y_), (x_+ w_, y_+h_), (0, 255, 0), 2)

            # Calculate landmarks from face detection and draw a circle
            shape = predictor(img[y:y+h, x:x+w], rect_faces[i])
            shape = face_utils.shape_to_np(shape)

            for (x1, y1) in shape:
                cv2.circle(img[y:y+h, x:x+w], (x1, y1), 2, (0, 0, 255), -1)
        else:
            # Draw a rectangle
            cv2.rectangle(img, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)

            # Calculate landmarks from face detection and draw a circle
            shape = predictor(img, rect_faces[i])
            shape = face_utils.shape_to_np(shape)

            for (x1, y1) in shape:
                cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)

    return img

def show_only_faces(faces, img, coords, bs):
    # Show all faces detected sequentially
    x, y, w, h = coords
    if bs:
        bbox = img[y:y + h, x:x + w]
        for face in faces:
            cv2.imshow('Bbox', bbox)
            cv2.imshow('Cara', face)
            cv2.waitKey(0)
    else:
        for face in faces:
            cv2.imshow('Cara', face)
            cv2.waitKey(0)

def face_feature_vector(faces, modelo):
    # Each face is coded by a numpy array depending the neural network model used
    embedding = []
    for cara in faces:
        embedding.append(np.array(represent_v2(cara, model_name = modelo)).reshape(1, -1))

    return embedding

def build_model(model_name):

    """
    This function builds a deepface model
    Parameters:
        model_name (string): face recognition or facial attribute model
            VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
            Age, Gender, Emotion, Race for facial attributes
    Returns:
        built deepface model
    """

    global model_obj #singleton design pattern

    models = {
        'VGG-Face': VGGFace.loadModel,
        'OpenFace': OpenFace.loadModel,
        'Facenet': Facenet.loadModel,
        'Facenet512': Facenet512.loadModel,
        'DeepFace': FbDeepFace.loadModel,
        'DeepID': DeepID.loadModel,
        'Dlib': DlibWrapper.loadModel,
        'ArcFace': ArcFace.loadModel,
        'Emotion': Emotion.loadModel,
        'Age': Age.loadModel,
        'Gender': Gender.loadModel,
        'Race': Race.loadModel
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
            #print(model_name," built")
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]

def preprocess_face_v2(img, target_size = (224, 224), grayscale = False, enforce_detection = True, detector_backend='opencv', align=True):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = img
    base_img = img.copy()

    # img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)

    # --------------------------

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # --------------------------

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------
    # resize image to expected shape

     #img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    # First resize the longer side to the target size
    # factor = target_size[0] / max(img.shape)

    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    if grayscale == False:
        # Put the base image in the middle of the padded image
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                     'constant')
    else:
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------

    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels

def represent_v2(img, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'opencv', align = True):

    """
    This function represents facial images as vectors.
    Parameters:
        img_path: exact image path, numpy array or based64 encoded images could be passed.
        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.
        model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.
            model = DeepFace.build_model('VGG-Face')
        enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
    Returns:
        Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """

    if model is None:
        model = build_model(model_name)

    #---------------------------------

    #decide input shape
    input_shape =  input_shape_x, input_shape_y= functions.find_input_shape(model)

    #detect and align
    img = preprocess_face_v2(img = img
    	, target_size=(input_shape_y, input_shape_x)
    	, enforce_detection = enforce_detection
    	, detector_backend = detector_backend
    	, align = align)

    #represent
    embedding = model.predict(img)[0].tolist()

    return embedding

def text_person(img, coords, coords_face, name_img_database, bs):
    x, y, w, h = coords
    x_, y_, w_, h_ = coords_face

    name = name_img_database.rsplit('/', 1)[1]
    name = name.rsplit('.', 1)[0]
    if bs:
        cv2.putText(img[y:y+h, x:x+w], name, (x_ + int(w_ / 2), y_ - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 111, 255), 2)
    else:
        cv2.putText(img, name, (x_ + int(w_ / 2), y_ - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 111, 255), 2)

def register_face(path_database, index, face, vector):
    directory = 'Persona_' + str(index)
    path = os.path.join(path_database[:-1], directory)
    os.mkdir(path)
    cv2.imwrite(path + '/P' + str(index) + '_0.png', face)

    f_logging = open(path + '/logging.txt', 'a+')
    timestr = time.strftime("%Y/%m/%d-%H:%M:%S") + '\n'
    f_logging.write(timestr)
    f_logging.close()

    np.save(path + '/vector' + str(index) + '_0.npy', vector)

def recognize_faces(img_db, name_img_db, vector_db, valid_faces, represent_faces, path_database,
                    next_person, frame_faces, coords, coords_faces, record_faces, bs, th_models):

    for k in range(len(represent_faces)):
        found = False
        for l in range(len(img_db)):
            dist = euclidean_distances(vector_db[l], represent_faces[k])
            if dist < th_models:
                if not found:
                    text_person(frame_faces, coords, coords_faces[k], name_img_db[l], bs)
                    found = True

                name = name_img_db[l].rsplit('/', 1)[0]
                f_logging = open(name + '/logging.txt', 'a+')
                timestr = time.strftime('%Y/%m/%d-%H:%M:%S') + '\n'
                f_logging.write(timestr)
                f_logging.close()
            elif record_faces:
                next_person += 1
                register_face(path_database, next_person, valid_faces[k], represent_faces[k])

    return next_person

def display_frames(bs, record_faces, record_video, frame, frameR, frame_rois, frame_faces, show_info, scale, fps):

    # Display state of record_faces flag
    height, width, channels = frame.shape
    height_, width_, channels_ = frame_faces.shape
    height_R, width_R, channels_R = frameR.shape

    if record_faces:
        text = 'Registering faces'

        cv2.putText(frame_faces, text, (width_ - int((170*scale)), int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (255, 0, 0), 1)
        cv2.circle(frame_faces, (width_ - int(20*scale), int(10*scale)), 8, (255, 0, 0), -1)

    if record_video:
        text = 'Recording video'
        cv2.putText(frame_faces, text, (width_ - int((170*scale)), int(40*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (0, 0, 255), 1)
        cv2.circle(frame_faces, (width_ - int(20*scale), int(35*scale)), 8, (0, 0, 255), -1)

        text_warning = 'WARNING: While recording rescaling is disabled'
        cv2.putText(frame_faces, text_warning, (width_ - int((380*scale)), int(height_ - 15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (0, 0, 255), 1)

        text_time = time.strftime('%H-%M-%S')
        cv2.putText(frameR, text_time, (int(5*scale), int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (0, 0, 255), 1)

    # Display state of show_info flag
    if show_info:
        infoLines = []
        infoLines.append('Cam resolution: ' + str(int(width)) + 'x' + str(int(height)))
        infoLines.append('Process resolution: ' + str(int(width_)) + 'x' + str(int(height_)) + ' @ ' + fps)

        posy = int(15*scale)
        for infoLine in infoLines:
           cv2.putText(frame_faces, infoLine, (int(5*scale), posy), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (100, 255, 0), 1, cv2.LINE_AA)
           posy += int(15*scale)

    # Display interesting frames
    if bs:
        #img_concat = cv2.hconcat([frameR, frame_rois, frame_faces])
        img_concat = frameR

    else:
        #img_concat = cv2.hconcat([frameR, frame_faces])
        img_concat = frameR

    cv2.imshow('Out', img_concat)
