import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from PIL import Image, ImageOps

def extract_and_save_activation_maps(model, sampled, maps_dir='maps'):
    os.makedirs(maps_dir, exist_ok=True)
    layer_outputs = [layer.output for layer in model.layers[:12]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    xx = sampled
    for i, x in enumerate(xx):
        activations = activation_model.predict(x.reshape(1, 28, 28, 1))
        first_layer_activation = activations[3]
        act = first_layer_activation[0, :, :, 0]
        fig = plt.figure()
        plt.imshow(act, cmap='gray')
        plt.axis('off')
        fig.savefig(f'{maps_dir}/activation{i}.png', bbox_inches='tight')
        plt.close(fig)
        img = Image.open(f'{maps_dir}/activation{i}.png').convert('L')
        img = ImageOps.fit(img, (28, 28))
        img.save(f'{maps_dir}/activation{i}.jpeg', 'JPEG')

def load_activation_images_to_array(maps_dir='maps', count=None):
    files = sorted([f for f in os.listdir(maps_dir) if f.endswith('.jpeg')])
    if count is not None:
        files = files[:count]
    pr = []
    for i, filename in enumerate(files):
        p = Image.open(os.path.join(maps_dir, filename)).convert('L')
        p = np.array(p)
        p = p.reshape(1, 28, 28, 1)
        pr = np.append(pr, p) if len(pr) else np.array([p])
    pr = pr.reshape(-1, 28, 28, 1)
    print('final', pr.shape)
    return pr
