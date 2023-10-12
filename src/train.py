import numpy as np
from network import NeuralNetwork
import dataset
import random
import pickle

train_images, train_labels = dataset.load_training()

input_images = [
    dataset.convert_image_to_input(image) for image in train_images
]

expected_labels = [
    dataset.convert_label_to_output(label) for label in train_labels
]

training_data = list(zip(input_images, expected_labels))

# ---------------------------------------

nn = NeuralNetwork([784, 72, 48, 10])
# with open('model_643216_2.bin', 'rb') as f:
#     nn = pickle.load(f)


def evaluate(test_data):
    correct = 0

    for test_image, test_label in test_data:
        result = nn.feed_forward(test_image)

        result_label = dataset.convert_output_to_label(result)
        expected_label = dataset.convert_output_to_label(test_label)

        if expected_label == result_label:
            correct += 1

    return correct


epochs = 10
lr = 0.001

num_tests = 10000

for epoch in range(epochs):
    for x, y in training_data:
        wg, bg = nn.backprop(x, y)
        nn.adjust(lr, wg, bg)

    correct = evaluate(training_data[:num_tests])
    print(f'Finished epoch #{epoch}: {correct}/{num_tests}')

    random.shuffle(training_data)

correct = evaluate(training_data)
print(f'Final accuracy: {(correct / len(training_data)):.2f}')

with open('model.bin', 'wb') as f:
    pickle.dump(nn, f)
