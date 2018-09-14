import sys
sys.path.insert(0, "/home/cariaggi/PythonPackages/")
sys.path.insert(0, "/home/cariaggi/lime-packages/")

import os
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from keras.applications import inception_v3
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import time
from functools import partial
from grid_segmentation import gridSegmentation
from evaluation_measures import relative_quality_of_explanation,absolute_quality_of_explanation
import clustering
from helper_functions import absoluteFilePaths
from multiprocessing import Pool, cpu_count

try:
    import lime
except:
    sys.path.append('..') # add the current directory
    import lime
from lime import lime_image, grid_segmentation, evaluation_measures, helper_functions, clustering

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

img = transform_img_fn(['../../chosen_1000_images/ILSVRC2012_val_00000309.JPEG'])
gt = load_ground_truths(['../../immagini_ritagliate/pack_2/ILSVRC2012_val_00000309.jpg'])

df = pd.read_csv("../../predizioni_bb.txt", " ")
bb_outcomes = list(df["class_number"])

# class = 'hotdog'
label = 934

model = inception_v3.InceptionV3()

explainer = lime_image.LimeImageExplainer(verbose=False)

start_time = time.time()

explanation = explainer.explain_instance(img[0], 
                                         model.predict, 
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=100)
end_time = time.time()
print("Explanation time: %s sec" % str(end_time - start_time))

try:
    positive_features = [area[0] for area in explanation.local_exp[label] if area[1] > 0]
except:
    print("KeyError: %s" % str(label))

tot_num_features = len(positive_features)
current_img_scores = []

start_time = time.time()

i=0

pool = Pool(processes=cpu_count()) 
multiple_res = [pool.apply_async(evaluation_measures.loop_body, 
                                 (i, 
                                  positive_features, 
                                  num_feat, 
                                  gt[i], 
                                  explanation.segments)) 
                for num_feat in range(1, tot_num_features+1)]

pool.close()
current_img_scores = [res.get(timeout=100) for res in multiple_res]
current_img_scores = sorted(current_img_scores, key=lambda x: x[0])
pool.join()
del pool
end_time = time.time()
#for num_feat in range(1, tot_num_features+1):
#        
#    # Features highlighted by the explanation. We take the top
#    # 'num_feat' positive features every time
#    highlighted_features = positive_features[:num_feat]
#    
#    #print len(highlighted_features)#, explanation.local_exp[label_to_exp]
#
#    
#    rel_quality = relative_quality_of_explanation(gt[0],
#                                               highlighted_features,
#                                               explanation.segments)
#    
#    abs_quality = absolute_quality_of_explanation(gt[0],
#                                               highlighted_features,
#                                               explanation.segments)
#    
#    current_img_scores.append((rel_quality, abs_quality, num_feat))


print("Evaluation time (all possible highlighted feats): %s sec" % str(end_time - start_time))
