# Уменьшение количества цветов изображения

import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
import skimage
import pylab

np.set_printoptions(linewidth=120, threshold=np.inf)

image = imread('data/parrots.jpg')

data = skimage.img_as_float(image)

pylab.imshow(data)
pylab.title('original')
pylab.show()

height_initial = len(data)
width_initial = len(data[0])

print('Size: %s x %s, %s pixels total' % (width_initial, height_initial, width_initial * height_initial))

rgb = np.reshape(data, (height_initial * width_initial, 3))

clf = KMeans(init='k-means++', random_state=241)
clf.fit(rgb)

print('N clusters:', clf.n_clusters)

data_by_clusters = clf.predict(rgb)

rgb_mean = []
for i in data_by_clusters:
    rgb_mean.append(clf.cluster_centers_[i])

rgb_mean = np.array(rgb_mean)

data_mean = np.reshape(rgb_mean, (height_initial, width_initial, 3))

pylab.figure(1)
pylab.imshow(data_mean)
pylab.title('mean')
pylab.show()

mse = np.square(np.subtract(rgb, rgb_mean)).mean()
psnr = 10 * np.log10([np.square(1) / mse])[0]

print('PSNR:', np.round(psnr, 0))

median_colors = []
for n in range(0, clf.n_clusters):
    selected = []
    selected[:] = data_by_clusters == n
    selected_pixels = rgb[selected]
    median_color = [np.median(selected_pixels[:,0]),np.median(selected_pixels[:,1]),np.median(selected_pixels[:,2])]
    median_colors.append(median_color)
    print('Cluster:', n, ', pixels selected:', len(selected_pixels), ', color:', median_color)

rgb_median = []
for i in data_by_clusters:
    rgb_median.append(median_colors[i])

rgb_median = np.array(rgb_median)

data_median = np.reshape(rgb_median, (height_initial, width_initial, 3))

pylab.figure(2)
pylab.imshow(data_median)
pylab.title('median')
pylab.show()

mse = np.square(np.subtract(rgb, rgb_median)).mean()
psnr = 10 * np.log10([np.square(1) / mse])[0]

print('PSNR:', np.round(psnr, 0))


for n in range(1,21):
    clf = KMeans(init='k-means++', random_state=241, n_clusters=n)
    clf.fit(rgb)
    data_by_clusters = clf.predict(rgb)
    rgb_new = []
    for i in data_by_clusters:
        rgb_new.append(clf.cluster_centers_[i])
    rgb_new = np.array(rgb_new)
    data_new = np.reshape(rgb_new, (height_initial, width_initial, 3))
    mse = np.square(np.subtract(rgb, rgb_new)).mean()
    psnr = 10 * np.log10([np.square(1) / mse])[0]
    print('N:', n, ', clusters:', clf.n_clusters, ', PSNR:', round(psnr,0))
    pylab.figure(n + 10)
    pylab.imshow(data_new)
    pylab.title(n)
    pylab.show()
