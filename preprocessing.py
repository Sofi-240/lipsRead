import numpy as np
import cv2
import tensorflow as tf


def extractFace(tf_rgb_frames):
    rgb_frames = tf_rgb_frames.numpy()

    image_median = np.median(
        rgb_frames, axis=0
    )

    image_hsv = cv2.cvtColor(
        image_median, cv2.COLOR_BGR2HSV
    )

    threshold, image_bin = cv2.threshold(
        image_hsv[:, :, 0].astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3), dtype=np.uint8
    )

    image_bin = cv2.morphologyEx(
        image_bin, cv2.MORPH_OPEN, kernel, iterations=1
    )

    r, c = np.where(
        image_bin == 1
    )

    boundary = [
        [
            c.min(), min([image_median.shape[1] - 1, c.max()])
        ],
        [
            r.min(), min([image_median.shape[0] - 1, r.max()])
        ],
    ]

    tf_gray_frames_crop = tf.image.rgb_to_grayscale(
        tf_rgb_frames[:, boundary[1][0]:boundary[1][1], boundary[0][0]:boundary[0][1], :]
    )

    image_median = np.median(
        tf_gray_frames_crop.numpy(), axis=0
    )

    face_cascade = cv2.CascadeClassifier(
        "data\\haarcascade_frontalface_alt.xml"
    )

    face = face_cascade.detectMultiScale(
        image_median.astype(np.uint8)
    )
    if type(face) is tuple:
        return tf_gray_frames_crop

    boundary = [
        [
            face[0, 0], min([image_median.shape[1] - 1, face[0, 0] + face[0, 2]])
        ],
        [
            face[0, 1], min([image_median.shape[0] - 1, face[0, 1] + face[0, 3]])
        ],
    ]
    tf_gray_frames_crop = tf_gray_frames_crop[:, boundary[1][0]:boundary[1][1], boundary[0][0]:boundary[0][1], :]
    return tf_gray_frames_crop


