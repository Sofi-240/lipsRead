import os
import cv2
import tensorflow as tf

vocab = list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
char2num = tf.keras.layers.StringLookup(
    vocabulary=vocab, oov_token=""
)
num2char = tf.keras.layers.StringLookup(
    vocabulary=char2num.get_vocabulary(), oov_token="", invert=True
)


def loadVideo(path):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, frame = cap.read()
        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )
        frames.append(
            tf.convert_to_tensor(
                frame
            )
        )
    cap.release()
    frames = tf.cast(
        frames, tf.float32
    )
    return frames


def loadAlignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    words = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            words = [*words, ' ', line[2]]
    words = char2num(
        tf.reshape(
            tf.strings.unicode_split(
                words, input_encoding='UTF-8'
            ), (-1)
        )
    )[1:]
    return words


def loadData(path):
    path = path.numpy().decode('utf-8')
    fileName = path.split('\\')[-1].split('.')[0]
    videoPath = os.path.join(
        'data', 's1', f'{fileName}.mpg'
    )
    alignmentPath = os.path.join(
        'data', 'alignments', 's1', f'{fileName}.align'
    )
    frames = loadVideo(videoPath)
    alignments = loadAlignments(alignmentPath)
    return frames, alignments


def mapData(path):
    ret = tf.py_function(
        loadData, [path], (tf.float32, tf.int64)
    )
    return ret


def createPipeline():
    def map_dir(txt):
        if txt.split('.')[-1] == 'mpg':
            return True
        return False

    files = os.listdir(
        './data/s1/'
    )

    files = list(
        filter(
            map_dir, files
        )
    )

    data = tf.data.Dataset.from_tensor_slices(files)

    data = data.shuffle(
        len(files), reshuffle_each_iteration=False
    )

    data = data.map(
        map_func=mapData
    )

    data = data.padded_batch(
        2, padded_shapes=([75, 288, 360, 3], [40])
    )

    data = data.prefetch(
        tf.data.AUTOTUNE
    )
    return data

