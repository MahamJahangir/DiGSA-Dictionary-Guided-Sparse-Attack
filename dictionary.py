import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import DictionaryLearning

def learn_dictionary(perturbations, n_components=100, alpha=1):
    print('Learning the dictionary...')
    t0 = time()
    dico = DictionaryLearning(n_components=n_components, alpha=alpha, transform_algorithm='omp')
    V = dico.fit(perturbations.reshape(perturbations.shape[0], -1)).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:n_components]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned \n' + 'Train time %.1fs on %d patches' % (dt, len(perturbations.reshape(perturbations.shape[0], -1))), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    return dico, V

def compute_sparse_reconstruction(dico, V, sampled):
    sampled_reshaped = sampled.reshape(sampled.shape[0], -1)
    Y = sampled_reshaped / 255
    t0 = time()
    x = dico.transform(Y)
    x = np.ravel(np.dot(x, V))
    print(x.shape)
    Y_flat = np.ravel(Y)
    print(Y_flat.shape)
    squared_error = np.sum((Y_flat - x) ** 2)
    print(squared_error)
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return x
