import numpy as np


def load_training():
    images_file = open('./train-images-idx3-ubyte', 'rb')
    labels_file = open('./train-labels-idx1-ubyte', 'rb')

    images = load_images(images_file)
    labels = load_labels(labels_file)

    return images, labels


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')


def load_images(file, num=60000):
    magic = bytes_to_int(file.read(4))
    if (magic != 2051):
        raise RuntimeError('Wrong file for images')

    num_images = bytes_to_int(file.read(4))
    num_rows = bytes_to_int(file.read(4))
    num_cols = bytes_to_int(file.read(4))

    images = []

    for i in range(min(num_images, num)):
        images.append([
            bytes_to_int(file.read(1)) for p in range(num_rows * num_cols)
        ])

    return images


def load_labels(file, num=60000):
    magic = bytes_to_int(file.read(4))
    if (magic != 2049):
        raise RuntimeError('Wrong file for labels')

    num_labels = bytes_to_int(file.read(4))

    labels = [
        bytes_to_int(file.read(1)) for l in range(min(num_labels, num))
    ]

    return labels


def convert_label_to_output(label: int):
    output = np.zeros((10, 1))
    output[label][0] = 1.0

    return output


def convert_image_to_input(image: list):
    return np.reshape(image, (784, 1))


def convert_output_to_label(output: np.ndarray[np.float64]):
    return np.argmax(output)
