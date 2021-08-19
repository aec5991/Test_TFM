from utils import *
import glob
import time
import dlib
import configparser
import random
import csv

config = configparser.ConfigParser()

config.read('configTest2.ini')
sections = config.sections()

import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def load_bs_algorithm(num, init_frames = 20):
    if num == 1:    # Mixture of Gaussians algorithm
        return cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    if num == 2:    # Exponential (adaptive) filter algorithm
        bs_init = []
        for i in range(init_frames):
            ret, frame = cap.read()
            bs_init.append(frame)

        bg = initialization_exp_filter(bs_init, init_frames)
        return bg

header = ['Benchmark', 'Resolution', 'Time BS', 'Time FD', 'Time FR', 'Time Total']
with open('Timing.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(header)
    for bench in range(31):
        random.shuffle(sections)
        for section in sections:
            num_section = 1

            num_frames = 1
            num_faces = 0
            new_frame_time = time.time()
            prev_frame_time = new_frame_time
            count_fps = 0
            fps = ''
            time_bs = 0
            time_fd = 0
            time_fr = 0
            time_total = 0
            resolutions = [(1920, 1080), (1280, 720)]
            for res in range(len(resolutions)):

                start_total = time.process_time()
                filename = config[section]['clip']
                cap = cv2.VideoCapture(filename)
                if (cap.isOpened()):
                    # We read first frame
                    ret, frame = cap.read()

                    # We get background subtraction flag
                    bs = config[section].getboolean('bs')

                    if bs:
                        # We choose background subtraction algorithm
                        bs_type = config[section].getint('bs_type')
                        fgbg = load_bs_algorithm(bs_type, init_frames = 20)

                    # We choose face detector algorithm
                    fd_type = config[section].getint('fd_type')
                    face_detector = load_face_detector(fd_type)

                    # We choose shape predictor for landmarks
                    predictor = dlib.shape_predictor(config[section]['predictor'])

                    # Neural networks models for represent each face as a feature vector
                    model = config[section]['models']
                    th_base = 0.55
                    th_models = config[section]['th_models']

                    # Load all database images and feature vector for recognize faces
                    path_database = config[section]['path_database']
                    format_img_database = config[section]['format_img_database']
                    img_db, name_img_db, vector_db, name_vector_db = load_images_from_database(path_database, format_img_database)
                    #print('Numero total de imagenes del database:', len(img_db))

                    # Keep track the number of faces
                    global next_person
                    next_person = 0
                    folders = glob.glob(path_database)
                    for folder in folders:
                        for f in glob.glob(folder):
                            num = int(f.rsplit('_', 1)[1])
                            if num > next_person:
                                next_person = num

                    record_faces = config[section].getboolean('record_faces')
                    record_video = False
                    show_info = config[section].getboolean('show_info')
                    scale = 1

                    th_models = config[section].getfloat('th_models')

                    # Iterate for each frame read
                    while (cap.isOpened() and ret):
                        # Frame preprocessing
                        #frameR = cv2.resize(frame, None, fx = scale, fy = scale, interpolation=cv2.INTER_CUBIC)
                        frameR = cv2.resize(frame, resolutions[res])

                        frame_gray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

                        frame_faces = frameR.copy()
                        frame_rois = frameR.copy()

                        if bs:
                            # Applying background subtraction algorithm chosen
                            start_bs = time.process_time()
                            if bs_type == 1:
                                fgmask = fgbg.apply(frame_gray)
                                #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, strel)
                            else:
                                fgmask, bg = exponentialFilter(frame = frame, alpha = 0.7, morph = True,
                                                               morph_kernel = (2, 2), background_init = fgbg)
                            time_bs += time.process_time() - start_bs

                            # Getting regions of interest for face detection
                            img_rois, roi = getROI(fgmask, config[section].getint('th_ROI'))
                            #print('Numero de rois:', len(rois))

                            # Processing each region of interest
                            for i in range(len(roi)):

                                # Draw ROI and getting its coordinates
                                frame_rois, coords = draw_ROI(roi[i], img_rois)

                                # Getting faces from ROI coordinates (ideally 1 per ROI)
                                start_fd = time.process_time()
                                detected_faces = face_detection(fd_type, face_detector, frame_gray, coords, bs)
                                time_fd += time.process_time() - start_fd

                                # Cleaning all faces returning those with correct shape and its coordinates with respect to the ROI
                                valid_faces, coords_faces, rect_faces = correction_face_detection(detected_faces, frame_faces, coords, fd_type, bs)

                                # If there is at least one valid face
                                if len(valid_faces) !=0:
                                    num_faces += 1

                                    # Draw faces and landmarks
                                    frame_faces = draw_faces_and_landmarks(coords, coords_faces, rect_faces, frame_faces, predictor, bs)

                                    # Visualize only bbox and individual faces
                                    #show_only_faces(valid_faces, frame, coords, bs)

                                    # Represent each face as a feature vector
                                    start_fr = time.process_time()
                                    represent_faces = face_feature_vector(valid_faces, model)
                                    time_fr += time.process_time() - start_fr

                                    # If database is empty, we initialize if we want
                                    if len(img_db) == 0 and record_faces:
                                        next_person = 0
                                        for m in range(len(valid_faces)):
                                            register_face(path_database, next_person, valid_faces[m], represent_faces[m])
                                        next_person = m

                                        img_db, name_img_db, vector_db, name_vector_db = load_images_from_database(path_database, format_img_database)

                                    # Recognizing people and logging. Also we can record new faces
                                    next_person = recognize_faces(img_db, name_img_db, vector_db, valid_faces,
                                    represent_faces, path_database, next_person, frame_faces, coords, coords_faces, record_faces, bs, th_models)

                        else:
                            # Getting faces from whole image
                            coords = (0, 0, 0, 0)
                            start_fd = time.process_time()
                            detected_faces = face_detection(fd_type, face_detector, frame_gray, coords, bs)
                            time_fd += time.process_time() - start_fd

                            # Cleaning all faces returning those with correct shape
                            valid_faces, coords_faces, rect_faces = correction_face_detection(detected_faces, frame_faces, coords, fd_type, bs)

                            # If there is at least one valid face
                            if len(valid_faces) != 0:
                                num_faces += 1

                                # Draw faces and landmarks
                                frame_faces = draw_faces_and_landmarks(coords, coords_faces, rect_faces, frame_faces, predictor, bs)

                                # Visualize only bbox and individual faces
                                #show_only_faces(valid_faces, frame, coords, bs)

                                # Represent each face as a feature vector
                                start_fr = time.process_time()
                                represent_faces = face_feature_vector(valid_faces, model)
                                time_fr += time.process_time() - start_fr

                                # If database is empty, we initialize if we want
                                if len(img_db) == 0 and record_faces:
                                    next_person = 0
                                    for m in range(len(valid_faces)):
                                        register_face(path_database, next_person, valid_faces[m], represent_faces[m])
                                    next_person = m

                                    img_db, name_img_db, vector_db, name_vector_db = load_images_from_database(path_database,
                                                                                                               format_img_database)

                                # Recognizing people and logging. Also we can record new faces
                                next_person = recognize_faces(img_db, name_img_db, vector_db, valid_faces,
                                                              represent_faces, path_database, next_person, frame_faces, coords,
                                                              coords_faces, record_faces, bs, th_models)

                        # Getting fps
                        count_fps += 1
                        new_frame_time = time.time()
                        if new_frame_time - prev_frame_time > 1:
                            fps = count_fps / (new_frame_time - prev_frame_time)
                            count_fps = 0
                            prev_frame_time = new_frame_time
                            fps = 'fps: ' + str(int(round(fps)))

                        # Frame steps
                        if num_frames % config[section].getint('steps_frame') == 0:
                            if record_video:
                                # Save results in video format
                                save_video.write(frame_faces)

                                text_time = time.strftime('%H-%M-%S')
                                cv2.putText(frameR, text_time, (int(5 * scale), int(15 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                                            (0, 0, 255), 1)
                                save_video_ori.write(frameR)

                        # Interrupt execution
                        key = cv2.waitKey(1) & 0xFF

                        # Keyboard processing
                        if key == ord('q'):
                            break
                        if key == ord('f'):
                            record_faces = not record_faces
                        if key == ord('r'):
                            record_video = not record_video
                            # We create output recording configuration
                            if record_video:
                                fps_record = 30
                                timestr = time.strftime('%H-%M-%S')
                                name = 'Rec_' + timestr + '.mp4'
                                save_video = config_video_save(frameR, fps_record, name)

                                name_ori = 'Ori_' + timestr + '.mp4'
                                save_video_ori = config_video_save(frameR, fps_record, name_ori)
                            else:
                                save_video.release()
                                save_video_ori.release()
                        if key == ord('i'):
                            show_info = not show_info
                        if key == ord('+'):
                            if not record_video:
                                if round(scale, 1) < 2:
                                    scale += 0.1
                        if key == ord('-'):
                            if not record_video:
                                if round(scale, 1) > 0.5:
                                    scale -= 0.1
                        if key == ord('='):
                            if not record_video:
                                scale = 1

                        display_frames(bs, record_faces, record_video, frame, frameR, frame_rois, frame_faces, show_info, scale, fps)

                        # Read next frame
                        ret, frame = cap.read()
                        num_frames += 1

                        # Reload new images and feature vector from database
                        img_db, name_img_db, vector_db, name_vector_db = load_images_from_database(path_database, format_img_database)

                cap.release()
                cv2.destroyAllWindows()

                time_total = time.process_time() - start_total

                print('Benchmark: ', bench)
                print('Seccion: ', num_section)
                print('Total caras detectadas: ', num_faces)

                print('Tiempo background subtraction :', time_bs, 's')
                print('Tiempo face detection :', time_fd, 's')
                print('Tiempo face recognition :', time_fr, 's')
                print('Tiempo total de ejecucion :', time_total, 's', '\n')

                if res == 0:
                    resolution = '1920x1080'
                else:
                    resolution = '1280x720'

                # We skip first execution times
                if bench != 0:
                    data = [section, resolution, time_bs, time_fd, time_fr, time_total]
                    writer.writerow(data)

            num_section += 1