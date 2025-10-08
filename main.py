import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import models

from data import load_mnist, preprocess_data
from model import build_model, train_model, save_and_load_weights
from sampling import build_digit_dict, sample_digits
from activations import extract_and_save_activation_maps, load_activation_images_to_array
from dictionary import learn_dictionary
from sparse_save import save_sparse_perturbations
from attack import create_adversarial_examples, compute_l2_stats

def main():
    # Load and preprocess
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()
    x_train, x_test, y_train, y_test = preprocess_data(x_train_raw, x_test_raw, y_train_raw, y_test_raw)

    # Build, train, save & reload model
    model = build_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    save_and_load_weights(model, 'my_checkpoint')
    predictions = model.evaluate(x_test, y_test)
    print("Accuracy on legitimate test examples: {}%", predictions)

    # Data sampling
    (X_train, Y_train), (X_test, Y_test) = __import__('tensorflow').keras.datasets.mnist.load_data()
    digitDict = build_digit_dict(X_test, Y_test)
    sampled = sample_digits(digitDict)
    sampled = sampled.reshape(-1, 784)
    plt.imshow(sampled[5010].reshape(28,28), cmap=plt.cm.gray)

    # Extracting and Saving Feature Maps
    model.load_weights('my_checkpoint')
    extract_and_save_activation_maps(model, sampled, maps_dir='maps')

    # Load activation images into array
    perturbations = load_activation_images_to_array(maps_dir='maps')

    # Learn dictionary and sparse representation
    dico, V = learn_dictionary(perturbations, n_components=100, alpha=1)

    sampled = sampled.reshape(-1, 28, 28, 1)
    Y = sampled / 255
    Y = Y.reshape(Y.shape[0], -1)
    x = dico.transform(Y)
    x = np.ravel(np.dot(x, V))
    print(x.shape)
    Y = np.ravel(Y)
    print(Y.shape)
    squared_error = np.sum((Y - x) ** 2)
    print(squared_error)

    tr = sampled.reshape(sampled.shape[0], -1)
    gamma = dico.transform(tr)
    print(gamma.shape)
    fig = plt.figure()
    plt.imshow(gamma[0:1].reshape(10,10), cmap=plt.cm.gray)
    plt.axis('off')
    fig.savefig('p.jpeg', bbox_inches='tight')

    print(V.shape)
    originalimage = np.dot(gamma, V)
    print(originalimage.shape)
    plt.imshow(originalimage[6].reshape(28,28), cmap=plt.cm.gray)

    # Save sparse perturbations
    save_sparse_perturbations(gamma, out_dir='100-1TSparsePerturbations')

    # Attack
    x_noisy_image = create_adversarial_examples(x_test, 'SparseAdversarialImageGeneration/100-1TSparsePerturbations/perturbation512.jpeg', epsilon=0.01)
    norm_arr = compute_l2_stats(x_test, x_noisy_image)

    # Attack Success Rate
    predictions_adv = model.evaluate(x_noisy_image, y_test)
    print("Accuracy on adversarial test examples: {}%", predictions_adv)
    predictions_legit = model.evaluate(x_test, y_test)
    print("Accuracy on legitimate test examples: {}%", predictions_legit)

if __name__ == '__main__':
    main()
