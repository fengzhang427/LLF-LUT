import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def tensor2im(image_tensor, visualize=False, video=False):
    image_tensor = image_tensor.detach()

    if visualize:
        image_tensor = image_tensor[:, 0:3, ...]

    if not video:
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, channel_axis=2)
        return {'PSNR': psnr, 'SSIM': ssim}
    else:
        raise NotImplementedError
