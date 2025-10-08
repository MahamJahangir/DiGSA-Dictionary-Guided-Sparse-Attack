import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def save_sparse_perturbations(gamma, out_dir='100-1TSparsePerturbations'):
    os.makedirs(out_dir, exist_ok=True)
    i = 1
    for x in gamma:
        fig = plt.figure()
        plt.imshow(x.reshape(10,10), cmap=plt.cm.gray)
        plt.axis('off')
        fig.savefig(f'{out_dir}/perturbation{str(i)}.jpeg', bbox_inches='tight')
        original_image = Image.open(f'{out_dir}/perturbation{str(i)}.jpeg')
        size = (28,28)
        fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
        fit_and_resized_image.save(f'{out_dir}/perturbation{str(i)}.jpeg')
        i = i + 1
