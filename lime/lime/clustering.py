import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from PIL import Image
import ntpath
import grid_segmentation
import math

def get_knee_point_value(values):
    y = values
    x = np.arange(0, len(y))

    index = 0
    max_d = -float('infinity')

    for i in range(0, len(x)):
        c = closest_point_on_segment(a=[x[0], y[0]], b=[x[-1], y[-1]],p=[x[i], y[i]])
        d = np.sqrt((c[0] - x[i])**2 + (c[1] - y[i])**2)
        if d > max_d:
            max_d = d
            index = i

    return index


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
#    if u < 0:
#        closest_point = a
#    elif u > 1:
#        closest_point = b
#    else:
#        cp_x = sx1 + u * x_delta
#        cp_y = sy1 + u * y_delta
#        closest_point = [cp_x, cp_y]
    cp_x = sx1 + u * x_delta
    cp_y = sy1 + u * y_delta
    closest_point = [cp_x, cp_y]

    return closest_point

def undersample_images(imgs, size):
    return [undersample_image(imgs[i],size) for i in range(len(imgs))]

def undersample_image(img, size):
    segments = grid_segmentation.gridSegmentation(size, img)
    
    coordinates = grid_segmentation.get_blocks_coordinates(segments)
    new_size = int(math.sqrt(len(coordinates)))
    undersampled = (np.zeros((new_size,new_size,3))).astype(float)

    reds, greens, blues = [], [], []
    
    for (tl,br) in coordinates:
        channel0 = np.mean(img[tl[0]:br[0], tl[1]:br[1], 0])
        channel1 = np.mean(img[tl[0]:br[0], tl[1]:br[1], 1])
        channel2 = np.mean(img[tl[0]:br[0], tl[1]:br[1], 2])
        
        reds.append(channel0)
        greens.append(channel1)
        blues.append(channel2)
        
    undersampled[:,:,0] = np.asarray(reds).reshape(new_size,new_size)
    undersampled[:,:,1] = np.asarray(greens).reshape(new_size,new_size)
    undersampled[:,:,2] = np.asarray(blues).reshape(new_size,new_size)
    
    return undersampled.astype(int)

#def resize_images(old_abs_paths, destination_folder):
#    new_paths = []
#    
#    for path in old_abs_paths:
#        name = ntpath.basename(path)
#        img = Image.open(path)
#        img = img.resize((224,224))
#        new_filename = destination_folder + name
#        img.save(new_filename)
#        new_paths.append(new_filename)
#    return new_paths

def load_images(paths_of_images):
    img_array = []
    for i in range(len(paths_of_images)):
        img = plt.imread(paths_of_images[i])
        
        # Handling grayscale images
        if len(img.shape) < 3:
            temp = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)
            temp[:,:,0] = temp[:,:,1] = temp[:,:,2] = img
            img = temp
            
        img_array.append(img)
    return img_array

def flatten_images(images):
    img_array = []
    for im in images:
        img_array.append(np.float32(im).ravel())
    return np.vstack(img_array)

def prepare_images_for_kmeans(paths_of_images):
    imgs = load_images(paths_of_images)
    return flatten_images(imgs)

def k_means(images, k, max_iter=1000, num_different_initial_labelings=2):
    
    clt = KMeans(n_clusters=k, 
                 n_init=num_different_initial_labelings, 
                 max_iter=max_iter, 
                 n_jobs=-1)
    
    clt.fit(images)
    return (clt.inertia_, clt.labels_, clt.cluster_centers_)
    
#    compactness, labels, centers = cv2.kmeans(images,
#                                          k,
#                                          None,
#                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0),
#                                          num_different_initial_labelings,
#                                          cv2.KMEANS_RANDOM_CENTERS
#                                          )    
#    return (compactness, labels, centers)
