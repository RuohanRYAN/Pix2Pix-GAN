import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from visdom import Visdom
import torch
from torchvision.utils import make_grid

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img

def display_image(array):
    array = array.squeeze()
    array = array.transpose(0,1)
    array = array.transpose(2,1)
    plt.imshow(array)

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
def rebuild_img(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy = np.transpose(image_numpy,(2,0,1))
    return image_numpy
def rebuild_grid(grid_tensor):
    tensor = grid_tensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
    return tensor

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main',port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
        self.images = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    def image(self,images,var_name):
        image = rebuild_grid(images)
        if(var_name not in self.images):
            self.images[var_name] = self.viz.images(image,env=self.env)
        else:
            self.viz.images(image, env=self.env,win=self.images[var_name])
        # a = rebuild_grid(images[0])
        # b = rebuild_grid(images[1])
        # self.viz.images(a,env=self.env)
        # self.vis.images(b,env=self.env)
