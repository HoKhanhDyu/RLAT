from PIL import Image
import numpy as np

def save_img_mnist(img, path, x, y, r):
    img = img.view(28, 28)
    img = img.cpu().detach().numpy()
    img = (img * 0.5) + 0.5
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    