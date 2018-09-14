import sys
sys.path.insert(0, "/home/cariaggi/PythonPackages/")
sys.path.insert(0, "/home/cariaggi/lime-packages/")

import numpy as np
from keras.applications import inception_v3
from keras.preprocessing import image
import pandas as pd
import os
import tensorflow as tf
import keras
from functools import partial
from grid_segmentation import gridSegmentation
import evaluation_measures
import clustering
from helper_functions import absoluteFilePaths

# Function used to preprocess images before feeding
# them to the inception_v3 net
def transform_img_fn(path_list):
    path_list = sorted(path_list)
    out = []
    for i in range(len(path_list)):
        img = image.load_img(path_list[i], target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inception_v3.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

def load_ground_truths(paths):
    ground_truths = []
    for i in range(len(paths)):
        gt = image.load_img(paths[i], target_size=(299, 299))
        ground_truths.append(np.array(gt))
    return ground_truths

# ------------------ limiting the memory usage of gpus -------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)
# ------------------------------------------------------------------------

model = inception_v3.InceptionV3()

#gt_paths = ["../../mini_test/ILSVRC2012_val_00002226.jpg",
#            "../../mini_test/ILSVRC2012_val_00004039.jpg"]
#im_paths = ["../../chosen_1000_images/ILSVRC2012_val_00002226.JPEG",
#            "../../chosen_1000_images/ILSVRC2012_val_00004039.JPEG"]

gt_paths = sorted( absoluteFilePaths("../../immagini_ritagliate/"), 
                   key=lambda x: os.path.basename(x) )
im_paths = sorted( absoluteFilePaths("../../chosen_1000_images") )

imgs_clustering = sorted( absoluteFilePaths('../../chosen_1000_images_resized') )

df = pd.read_csv("../../predizioni_bb.txt", " ")
bb_outcomes = list(df["class_number"])

imgs = transform_img_fn(im_paths)
gts = load_ground_truths(gt_paths)

grid_sizes = [4,8,16,32,64]

lime_version = str(sys.argv[1])

print "explainer: %s" % lime_version

# Testing original version of Lime ------------------------------------------------------------
if lime_version == 'lime':
    hide_colors = [(0,'gray')]#[(-1,'black'), (0,'gray'), (1,'white')]
    for h in hide_colors:
        qualities = evaluation_measures.evaluate_explanations(lime_version,
                                                              model,
                                                              bb_outcomes,
                                                              imgs,
                                                              gts,
                                                              neigh_size=100,
                                                              segmentation_fun=None,
                                                              hide_col=h[0])
        filename = "exp_results/%s_hidecol=%s_neighsize=200" % (lime_version, h[1])
        np.save(filename, qualities)
# ---------------------------------------------------------------------------------------------

# Testing color version of Lime ---------------------------------------------------------------
elif lime_version == 'limecolor':
    hide_colors = [(None,'None')]#[(-1,'black'), (0,'gray'), (1,'white')]
    for h in hide_colors:
        qualities = evaluation_measures.evaluate_explanations(lime_version,
                                                              model,
                                                              bb_outcomes,
                                                              imgs,
                                                              gts,
                                                              neigh_size=100,
                                                              segmentation_fun=None,
                                                              hide_col=h[0])
        filename = "exp_results/%s_hidecol=%s_neighsize=200" % (lime_version, h[1])
        np.save(filename, qualities)
# ---------------------------------------------------------------------------------------------
        
# Testing Lime# -------------------------------------------------------------------------------
elif lime_version == 'lime#':
    lime_version = 'lime#color'
    hide_colors = [(None,'None')]#[(-1,'black'), (0,'gray'), (1,'white')]
    for g in grid_sizes:
        for h in hide_colors:
            qualities = evaluation_measures.evaluate_explanations(
                    lime_version,
                    model,
                    bb_outcomes,
                    imgs,
                    gts,
                    neigh_size=100,
                    segmentation_fun=partial(gridSegmentation,g),
                    hide_col=h[0])
            filename = \
                "exp_results/%s_hidecol=%s_gridsize=%s_neighsize=200" % (lime_version, h[1], g)
            np.save(filename, qualities)
# ---------------------------------------------------------------------------------------------

# Testing Lime# Random ------------------------------------------------------------------------
elif lime_version == 'lime#R':
    image_pool = imgs
    for g in [64]:
        qualities = evaluation_measures.evaluate_explanations(
                                              'lime#R',
                                              model,
                                              bb_outcomes,
                                              imgs,
                                              gts,
                                              neigh_size=100,
                                              segmentation_fun=partial(gridSegmentation,g),
                                              image_pool=image_pool)
        lime_version = 'lime#R-NEW'
        filename = "exp_results/%s_gridsize=%s_neighsize=200" % (lime_version, g)
        np.save(filename, qualities)
        
    del image_pool
# ---------------------------------------------------------------------------------------------

# Testing Lime# Clustering --------------------------------------------------------------------
elif lime_version == 'lime#C':
    ks_for_each_grid_size = [20,17,16,26,18]
    images = clustering.load_images(imgs_clustering)
    
    for k,g in zip(ks_for_each_grid_size, grid_sizes):
        und = clustering.undersample_images(images, g)
        flattened = clustering.flatten_images(und)
        inertia, lbls, centers = clustering.k_means(flattened, k)
        
        qualities = evaluation_measures.evaluate_explanations(
                                              'lime#C',
                                              model,
                                              bb_outcomes,
                                              imgs,
                                              gts,
                                              neigh_size=100,
                                              segmentation_fun=partial(gridSegmentation,g),
                                              clustering_labels=lbls)
        lime_version = 'lime#C-NEW'
        filename = "exp_results/%s_gridsize=%s_neighsize=200" % (lime_version, g)
        np.save(filename, qualities)
    
    del images
# ---------------------------------------------------------------------------------------------

else:
    print("Unsupported explainer type")

#print(qualities)

