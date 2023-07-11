import numpy as np
import cv2
import tensorflow as tf
import random


def cropByMouth(tf_rgb_frames):
    rgb_frames = tf_rgb_frames.numpy()

    image_median = np.median(
        rgb_frames, axis=0
    )

    image_hsv = cv2.cvtColor(
        image_median, cv2.COLOR_RGB2HSV
    )

    threshold, image_bin = cv2.threshold(
        image_hsv[:, :, 0].astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5), dtype=np.uint8
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

    tf_rgb_frames_crop = tf_rgb_frames[:, boundary[1][0]:boundary[1][1], boundary[0][0]:boundary[0][1], :]

    tf_gray_frames_crop = tf.image.rgb_to_grayscale(
        tf_rgb_frames_crop
    )
    image_median = np.median(
        tf_gray_frames_crop.numpy(), axis=0
    )

    face = faceCascade(
        contrast_stretch(image_median)
    )

    face_prop = [
        face[0][1] - face[0][0],
        face[1][1] - face[1][0]
    ]

    sample_index = random.sample(
        list(
            range(tf_gray_frames_crop.shape[0])
        ), 10
    )

    for smp in sample_index:
        curr_frame = contrast_stretch(
            tf_gray_frames_crop[smp, :, :, :].numpy()
        )
        curr_face = faceCascade(
            curr_frame
        )
        curr_prop = [
            curr_face[0][1] - curr_face[0][0],
            curr_face[1][1] - curr_face[1][0]
        ]

        if (curr_face[0][0] > 0 and curr_face[0][1] < curr_frame.shape[1] - 1 and curr_prop[0] > face_prop[0]) or (
                face_prop[0] == curr_frame.shape[1] - 1):
            face[0] = curr_face[0]
            face_prop[0] = curr_prop[0]

        if (curr_face[1][0] > 0 and curr_face[1][1] < curr_frame.shape[0] - 1 and curr_prop[1] > face_prop[1]) or (
                face_prop[1] == curr_frame.shape[0] - 1):
            face[1] = curr_face[1]
            face_prop[1] = curr_prop[1]

    tf_gray_frames_crop = tf_gray_frames_crop[:, face[1][0]:face[1][1], face[0][0]:face[0][1], :]
    tf_rgb_frames_crop = tf_rgb_frames_crop[:, face[1][0]:face[1][1], face[0][0]:face[0][1], :]

    image_median = np.median(
        tf_gray_frames_crop.numpy(), axis=0
    )

    mouth = mouthCascade(
        contrast_stretch(image_median)
    )

    mouth_prop = [
        mouth[0][1] - mouth[0][0],
        mouth[1][1] - mouth[1][0]
    ]

    sample_index = random.sample(
        list(
            range(tf_gray_frames_crop.shape[0])
        ), 10
    )

    for smp in sample_index:
        curr_frame = contrast_stretch(
            tf_gray_frames_crop[smp, :, :, :].numpy()
        )
        curr_mouth = mouthCascade(
            curr_frame
        )
        curr_prop = [
            curr_mouth[0][1] - curr_mouth[0][0],
            curr_mouth[1][1] - curr_mouth[1][0]
        ]

        if (curr_mouth[0][0] > 0 and curr_mouth[0][1] < curr_frame.shape[1] - 1 and curr_prop[0] > mouth_prop[0]) or (
                mouth_prop[0] == curr_frame.shape[1] - 1):
            mouth[0] = curr_mouth[0]
            mouth_prop[0] = curr_prop[0]

        if (curr_mouth[1][0] > curr_frame.shape[0] // 2 and curr_prop[1] > mouth_prop[1]) or (
                mouth_prop[1] == (curr_frame.shape[0] // 2) - 1):
            mouth[1] = curr_mouth[1]
            mouth_prop[1] = curr_prop[1]

    tf_gray_frames_crop = tf_gray_frames_crop[:, mouth[1][0]:mouth[1][1], mouth[0][0]:mouth[0][1], :]
    tf_rgb_frames_crop = tf_rgb_frames_crop[:, mouth[1][0]:mouth[1][1], mouth[0][0]:mouth[0][1], :]
    return tf_rgb_frames_crop, tf_gray_frames_crop


def contrast_stretch(img_gray):
    img_gray = 255 * ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()))
    return img_gray


def faceCascade(img_gray):
    cascade = cv2.CascadeClassifier(
        "data\\haarcascade_frontalface_default.xml"
    )

    rect = cascade.detectMultiScale(
        img_gray.astype(np.uint8)
    )

    if type(rect) == tuple:
        bound = [
            [
                0, img_gray.shape[1] - 1
            ],
            [
                0, img_gray.shape[0] - 1
            ]
        ]
        return bound
    bound = [
        [
            rect[0, 0], min([img_gray.shape[1] - 1, rect[0, 0] + rect[0, 2]])
        ],
        [
            rect[0, 1], min([img_gray.shape[0] - 1, rect[0, 1] + rect[0, 3] + 10])
        ],
    ]
    return bound


def mouthCascade(img_gray):
    cascade = cv2.CascadeClassifier(
        "data\\haarcascade_mcs_mouth.xml"
    )

    rect = cascade.detectMultiScale(
        img_gray.astype(np.uint8), 1.4
    )

    mid = img_gray.shape[0] // 2

    bound = [
        [
            0, img_gray.shape[1] - 1
        ],
        [
            mid, img_gray.shape[0] - 1
        ]
    ]

    if type(rect) == tuple:
        return bound

    rect = rect[rect[:, 1] > mid, :]

    if rect.shape[0] == 0:
        return bound

    if rect.shape[0] > 1:
        indices = np.argsort(rect[:, 2])[::-1]
        rect = np.reshape(
            rect[indices[0], :], (1, 4)
        )

    bound = [
        [
            max([0, rect[0, 0] - 10]), min([img_gray.shape[1] - 1, rect[0, 0] + rect[0, 2] + 10])
        ],
        [
            max([mid, rect[0, 1] - 10]), min([img_gray.shape[0] - 1, rect[0, 1] + rect[0, 3] + 10])
        ],
    ]
    return bound


def opticalFlowFeature(tf_array):
    tf_np_array = tf_array.numpy()

    ret_array = np.zeros_like(tf_np_array)

    for i in range(1, tf_np_array.shape[0]):
        flow = cv2.calcOpticalFlowFarneback(
            tf_np_array[i - 1][:, :, 0].astype(np.uint8),
            tf_np_array[i][:, :, 0].astype(np.uint8),
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=5,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )
        mag, _ = cv2.cartToPolar(
            flow[..., 0], flow[..., 1]
        )
        ret_array[i][:, :, 0] = cv2.normalize(
            mag, None, 0, 255, cv2.NORM_MINMAX
        )

    ret_array = tf.convert_to_tensor(
        ret_array, dtype=tf_array.dtype
    )

    return ret_array
