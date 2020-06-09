from diplom import Net, mean, std
import torch
from torch.autograd import Variable
import argparse
from skimage.measure import compare_ssim as ssim
import math
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import sys
import os
import PyQt5.uic as uic
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap
import cv2

os.makedirs("images/outputs", exist_ok=True)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

    @pyqtSlot(name='on_pushButton_clicked')
    def _start_experiment(self):
        ui = self.ui
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File',
                                                  os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop'))
        end = filename.split(".")[-1]
        if end != "jpg" and end != "png":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText("Выберите файл jpg или png")
            msg.setWindowTitle("Сообщение")
            msg.exec()
        else:
            path_lr, path_hr, img_start = neero(filename)
            map1 = QPixmap(path_hr)
            ui.label_8.setPixmap(map1)
            map2 = QPixmap(path_lr)
            ui.label_2.setPixmap(map2)
            target = cv2.imread(path_hr)
            ref = cv2.imread(path_lr)
            res = compare_images(target, ref)
            ui.label_4.setText('psnr: {:.2f}\nmse: {:.2f}\nssim: {:.2f}'.format(res[0], res[1], res[2]))

def psnr(target, ref):
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err


# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel=True))

    return scores


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def neero(dirimg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model and load model checkpoint
    netcon = Net().to(device)
    netcon.load_state_dict(torch.load('./saved_models/generator_300.pth'), strict=False)
    netcon.eval()
    size = Image.open(dirimg).size
    transform_lr = transforms.Compose([transforms.Resize((255 // 3, 255 // 3), Image.BICUBIC), transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    transform_hr = transforms.Compose([transforms.Resize((340,  340), Image.BICUBIC), transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    # Prepare input
    image_tensor = Variable(transform_lr(Image.open(dirimg).convert('RGB'))).to(device).unsqueeze(0)
    image_start = Variable(transform_hr(Image.open(dirimg).convert('RGB'))).to(device).unsqueeze(0)

    # Upsample image
    with torch.no_grad():
        sr_image = netcon(image_tensor)

    # Save image
    imgs_hr = torch.nn.functional.interpolate(image_tensor, scale_factor=4)

    gen_hr = make_grid(imgs_hr, nrow=1, normalize=True)
    img_first = make_grid(image_start, nrow=1, normalize=True)
    imgs_lr = make_grid(sr_image, nrow=1, normalize=True)
    save_image(imgs_lr, "./images/outputs/img_lr.png")
    save_image(gen_hr, "./images/outputs/img_hr.png")
    save_image(img_first, "./images/outputs/img_start.png")
    return os.curdir + "/images/outputs/img_lr.png", os.curdir + "/images/outputs/img_hr.png", os.curdir + "/images/outputs/img_start.png"


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    # neero('./22.jpg')
