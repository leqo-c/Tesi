#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:28:29 2018

@author: leo
"""

import copy

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb

try: 
    from . import lime_base
    from .wrappers.scikit_image import SegmentationAlgorithm
except:
    import lime_base
    from wrappers.scikit_image import SegmentationAlgorithm

class LimeImageTrivialExplainer(object):

    def __init__(self, kernel_width=.25, verbose=False,
                 feature_selection='auto', random_state=None):

        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         return_sample_neighborhood_images=False):

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        if return_sample_neighborhood_images:
            labels, sam = self.data_labels(image, fudged_image, segments,
            		                                    classifier_fn, num_samples,
            		                                    batch_size=batch_size,
            		                                    return_sample_neighborhood_images=return_sample_neighborhood_images)
        else:
            labels = self.data_labels(image, fudged_image, segments,
            		                                    classifier_fn, num_samples,
            		                                    batch_size=batch_size,
            		                                    return_sample_neighborhood_images=return_sample_neighborhood_images)


        if top_labels:
            orig_img_probs = classifier_fn(image)
            top = np.argsort(orig_img_probs)[-top_labels:]

        for label in top:
            target_class_probs = labels[:, label]
            
            
            
        if(return_sample_neighborhood_images):
            return ret_exp, sam
        return ret_exp
    
    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    batch_size=10,
                    return_sample_neighborhood_images=False,
                    fudged_images_pool=[]):

        import random
        
        if len(fudged_images_pool) == 0:
            fudged_images_pool = [fudged_image]
        
        n_features = np.unique(segments).shape[0]
        
        data = np.ones((n_features, n_features), dtype=int)
        for i in range(len(n_features)):
            data[i,i] = 0
            
        labels = []
        imgs = []
        #imgs_to_return = []
        samples = []
        
        for j in range(len(data)):
            row = data[j]
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape)
            
            fudged_images_indexes = range(1,len(fudged_images_pool)+1)
            
            for z in zeros:
                val = random.choice(fudged_images_indexes)
                mask[segments == z] = val
            
            for i in fudged_images_indexes:
                temp[mask == i] = fudged_images_pool[i-1][mask == i]
                
            imgs.append(temp)
            samples = imgs
            
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                #if(return_sample_neighborhood_images):
                #imgs_to_return.append(imgs)
                imgs = []
                
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        if(return_sample_neighborhood_images):
            return np.array(labels), samples
        else:
            return np.array(labels)
        
    def best_n_feats(self,
                   probs,
                   label,
                   num_feats):
        
        sorted_probs = sorted(probs, key=lambda x: x[0])
        return np.array(sorted_probs)[:,1][num_feats]
        
        
        