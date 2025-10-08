import numpy as np
from PIL import Image
import statistics

def create_adversarial_examples(x_test, perturbation_path, epsilon=0.01):
    x_noisy_image = []
    i = 1
    for x in x_test:
        p = Image.open(perturbation_path).convert('L')
        p = np.array(p)
        x = x.reshape(1, 28, 28, 1)
        p = p.reshape(1, 28, 28, 1)
        noisy_image = x + epsilon * p
        x_noisy_image = np.append(x_noisy_image, noisy_image) if len(x_noisy_image) else np.array([noisy_image])
        i = i + 1
    x_noisy_image = x_noisy_image.reshape(x_test.shape[0], 28, 28, 1)
    print('final', x_noisy_image.shape)
    return x_noisy_image

def compute_l2_stats(x_test, x_noisy_image):
    norm_arr = []
    for i in range(len(x_test)):
        l2 = np.linalg.norm(x_test[i:i+1] / 255 - x_noisy_image[i:i+1] / 255)
        norm_arr = np.append(norm_arr, l2) if len(norm_arr) else np.array([l2])
    print("mean", statistics.mean(norm_arr))
    print("median", statistics.median(norm_arr))
    return norm_arr
