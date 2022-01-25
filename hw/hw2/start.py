"""
Homework 2: Working with images in Torch
Author: Varun Aggarwal
Last Modified: 23 Jan 2022
"""


import torch
import torchvision.transforms as tvt

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


# Setup Data
data_path = "images"
image1_name = "img1.jpg"
image2_name = "img2.jpg"


# load images wiht PIL
img1 = Image.open(os.path.join(data_path, image1_name))
img2 = Image.open(os.path.join(data_path, image2_name))


# setup transform
transform_img = tvt.Compose(
    [
        tvt.PILToTensor(),
        tvt.ConvertImageDtype(torch.float),
        tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

# create image tensor
img1_tensor = transform_img(img1)
img2_tensor = transform_img(img2)


# create histogram
def get_hist(tensor_img, bins=100):
    return list(
        map(
            lambda x: torch.histc(tensor_img[x, :, :], bins=bins, min=-1.0, max=1.0),
            [0, 1, 2]
        )
    )


# get histogram from each image
bins = 100
hist_img1 = get_hist(img1_tensor, bins)
hist_img2 = get_hist(img2_tensor, bins)


# plot images and thier respective histograms
plt.figure(figsize=(15, 8))
X_axis = np.linspace(-0.5, 0.5, bins)
colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
for i in range(1, 4):
    plt.subplot(2, 4, i)
    plt.bar(X_axis, hist_img1[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(2, 4, i + 4)
    plt.bar(X_axis, hist_img2[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)

plt.subplot(2, 4, 4)
plt.imshow(img1)

plt.subplot(2, 4, 8)
plt.imshow(img2)

plt.savefig("orginalHistogram.png")


# apply affine transformation - https://pytorch.org/vision/master/auto_examples/plot_transforms.html#randomaffine
affine_transfomer = tvt.RandomAffine(
    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75), fill=-2.0
)
img1_tensor_affine = affine_transfomer(img1_tensor)
img2_tensor_affine = affine_transfomer(img2_tensor)


# apply perspective transformation -  https://pytorch.org/vision/master/auto_examples/plot_transforms.html#randomaffine
perspective_transformer = tvt.RandomPerspective(distortion_scale=0.6, p=1.0, fill=-2.0)
img1_tensor_perspective = perspective_transformer(img1_tensor)
img2_tensor_perspective = perspective_transformer(img2_tensor)


# plot images after affine and their respective histograms
plt.figure(figsize=(15, 16))
X_axis = np.linspace(-0.5, 0.5, bins)
colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
for i in range(1, 4):
    plt.subplot(4, 4, i)
    plt.bar(X_axis, hist_img1[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 4)
    plt.bar(X_axis, get_hist(img1_tensor_affine)[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 8)
    plt.bar(X_axis, hist_img2[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 12)
    plt.bar(X_axis, get_hist(img2_tensor_affine)[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)

plt.subplot(4, 4, 4)
plt.imshow(img1)
plt.subplot(4, 4, 8)
plt.imshow((img1_tensor_affine.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(4, 4, 12)
plt.imshow(img2)
plt.subplot(4, 4, 16)
plt.imshow((img2_tensor_affine.permute(1, 2, 0) + 1.0) / 2.0)

plt.savefig("affineHistogram.png")


# plot images after perspective and their respective histograms
plt.figure(figsize=(15, 16))
X_axis = np.linspace(-0.5, 0.5, bins)
colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
for i in range(1, 4):
    plt.subplot(4, 4, i)
    plt.bar(X_axis, hist_img1[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 4)
    plt.bar(X_axis, get_hist(img1_tensor_perspective)[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 8)
    plt.bar(X_axis, hist_img2[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)
    plt.subplot(4, 4, i + 12)
    plt.bar(X_axis, get_hist(img2_tensor_perspective)[i - 1], color=colors[i - 1])
    plt.ylim(0, 5e5)

plt.subplot(4, 4, 4)
plt.imshow(img1)
plt.subplot(4, 4, 8)
plt.imshow((img1_tensor_perspective.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(4, 4, 12)
plt.imshow(img2)
plt.subplot(4, 4, 16)
plt.imshow((img2_tensor_perspective.permute(1, 2, 0) + 1.0) / 2.0)

plt.savefig("perspectiveHistogram.png")


# Display Image with scaling
plt.subplot(2, 3, 1)
plt.imshow((img1_tensor.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(2, 3, 2)
plt.imshow((img1_tensor_affine.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(2, 3, 3)
plt.imshow((img1_tensor_perspective.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(2, 3, 4)
plt.imshow((img2_tensor.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(2, 3, 5)
plt.imshow((img2_tensor_affine.permute(1, 2, 0) + 1.0) / 2.0)
plt.subplot(2, 3, 6)
plt.imshow((img2_tensor_perspective.permute(1, 2, 0) + 1.0) / 2.0)

plt.savefig("allTransforms.png")


# calculate distances between histograms
def get_dist(hist1, hist2):
    return list(map(lambda x: wasserstein_distance(hist1[x], hist2[x]), [0, 1, 2]))


dist_without_transform = get_dist(hist_img1, hist_img2)
dist_with_affine = get_dist(get_hist(img1_tensor_affine), get_hist(img2_tensor_affine))
dist_with_perspective = get_dist(
    get_hist(img1_tensor_perspective), get_hist(img2_tensor_perspective)
)

print(dist_without_transform)
print(dist_with_affine)
print(dist_with_perspective)
