import itertools
import numpy as np
import lime_image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def evaluate_explanations(explainer_type,
                          model,
                          bb_outcomes,
                          images,
                          ground_truths,
                          neigh_size=100,
                          segmentation_fun=None,
                          hide_col=0,
                          image_pool=[], # For lime#R
                          clustering_labels=None, # For lime#C
                          shown_features=10000,
                          draw_prob=0.5): 
    
    """
    Evaluates the quality of the explanations of the given explainer
    on a set of images
    
    Args:
        explainer:         type of the explainer whose explanations 
                           have to be evaluated;
        model:             the black box model;
        bb_outcomes:       predictions of the black box;        
        images:            images in which we want to evaluate explanations
        ground_truths:     cut images, used as oracle when evaluating
                           explanations;
        neigh_size:        number of images generated in the neighborhood
                           of each image;
        segmentation_fun:  defines the shape of the explanation's features;
        hide_col:          when using basic versions of Lime, this is the
                           color corresponding to a turned-off feature;
        shown_features:    number of features shown in the explanation;
        draw_prob:         probability to extract an image of the same
                           cluster (just for LIME#RC)
    
    """
    lime_sharp_clus = False
    
    if explainer_type == 'lime' or explainer_type == 'lime#' \
    or explainer_type == 'limecolor' or explainer_type == 'lime#color':
        explainer = lime_image.LimeImageExplainer()
    elif explainer_type == 'lime#R':
        if len(image_pool) == 0:
            raise "Given image pool is not properly defined"
        #explainer = lime_image.LimeImagePatchworkExplainer(image_pool=image_pool)
        explainer = lime_image.LimeImageEnhancedPatchworkExplainer(image_pool=image_pool)
    elif explainer_type == 'lime#C':
        lime_sharp_clus = True
        explainer = lime_image.LimeImagePatchworkExplainer(image_pool=[])
        #explainer = lime_image.LimeImageEnhancedPatchworkExplainer(image_pool=[])
    elif explainer_type == 'lime#RC':
        lime_sharp_rc = True
    else:
        print("Unsupported explainer type")
        return
        
    
    # Each entry of this list consist of a pair (rel, abs) where
    # rel = relative quality of the image's explanation
    # abs = absolute quality of the image's explanation
    list_of_qualities = []
    
    for i in range(len(images)):
        
        # Set the proper image pool to draw the images from
        # (only for Lime#C)
        if lime_sharp_clus:
            img_pool = get_images_of_same_cluster(clustering_labels[i],
                                              clustering_labels,
                                              images)
            #print "len of pool = %d" % len(pool)
            explainer.image_pool = img_pool
            
        elif lime_sharp_rc:
            same_clus = get_images_of_same_cluster(clustering_labels[i], clustering_labels, images)
            other_clus = get_images_of_other_clusters(clustering_labels[i],
                                                      clustering_labels,
                                                      images)
            explainer = lime_image.LimeImageMixedPatchworkExplainer(same_clus, other_clus, draw_prob)
        
        # Explanation of the i-th image, using the previously
        # instantiated explainer
        explanation = explainer.explain_instance(images[i],
                                                 model.predict,
                                                 top_labels=1,
                                                 hide_color=hide_col,
                                                 num_samples=neigh_size,
                                                 segmentation_fn=segmentation_fun)
        
        # Black box prediction of the i-th image
        label_to_exp = bb_outcomes[i]
        
        try:
            positive_features = \
                    [area[0] for area in explanation.local_exp[label_to_exp] if area[1] > 0]
        except:
            print("KeyError: %s" % str(label_to_exp))
            continue
        
        tot_num_features = len(positive_features)
        current_img_scores = []
        
        pool = Pool(processes=cpu_count())
        multiple_res = [pool.apply_async(loop_body, 
                                         (positive_features, 
                                          num_feat, 
                                          ground_truths[i], 
                                          explanation.segments)) 
                        for num_feat in range(1, tot_num_features+1)]
        pool.close()
        current_img_scores = [res.get(timeout=100) for res in multiple_res]
        pool.join()
        del pool
        
        list_of_qualities.append(current_img_scores)
        
        if lime_sharp_clus:
            del img_pool
            
        if lime_sharp_rc:
            del same_clus
            del other_clus
        
    return np.array(list_of_qualities)

def loop_body(pos_feats, num_feats, gt, segments):
    highlighted = pos_feats[:num_feats]
    rel = relative_quality_of_explanation(gt, highlighted, segments)
    abso = absolute_quality_of_explanation(gt, highlighted, segments)
    return (rel, abso, num_feats)

def get_images_of_same_cluster(image_label, all_labels, all_images):
    
    all_labels = np.array(all_labels)
    indexes_of_images_in_the_same_cluster, = np.where(all_labels == image_label)
    all_images = np.array(all_images)
    result = all_images[indexes_of_images_in_the_same_cluster]
    
    return list(result)

def get_images_of_other_clusters(image_label, all_labels, all_images):
    all_labels = np.array(all_labels)
    indexes_of_images_in_other_clusters, = np.where(all_labels != image_label)
    all_images = np.array(all_images)
    result = all_images[indexes_of_images_in_other_clusters]
    
    return list(result)

def absolute_quality_of_explanation(ground_truth_image, 
                           highlighted_features,
                           segments):
    green_pixels_in_gt = 0.0
    tot_pixels_in_gt = 0.0
    
    # Count the number of both pixels and GREEN pixels of the ground truth
    # image, not considering white pixels. 
    for i,j in itertools.product(range(segments.shape[0]), range(segments.shape[1])):
        if not np.array_equal(ground_truth_image[i,j], [255,255,255]):
            tot_pixels_in_gt = tot_pixels_in_gt + 1
            if segments[i,j] in highlighted_features:
                green_pixels_in_gt = green_pixels_in_gt + 1

    # Then, return the ratio between the two values as a measure of quality                
    return green_pixels_in_gt / tot_pixels_in_gt
    

def relative_quality_of_explanation(ground_truth_image, 
                           highlighted_features,
                           segments):
    # Represents the green areas that cover (at least half) a part
    # of the ground truth image
    good_features = 0.0
    
    for feat in highlighted_features:
        # feat is a feature, so either a superpixel or
        # a square block
        feat_pixels_in_gt, feat_green_pixels_in_gt = 0.0, 0.0
        
        for i,j in itertools.product(range(segments.shape[0]), range(segments.shape[1])):
            # compute the percentage of pixels in the feature
            # that intersect with the ground truth image
            if segments[i,j] == feat:
                feat_pixels_in_gt = feat_pixels_in_gt + 1
                if not np.array_equal(ground_truth_image[i,j], [255,255,255]):
                    feat_green_pixels_in_gt = feat_green_pixels_in_gt + 1
        if feat_pixels_in_gt > 0:
            ratio = feat_green_pixels_in_gt / feat_pixels_in_gt
        else:
            ratio = 0
        
        # if the percentage of pixels covered by the feature is
        # higher than 50%, then we say the feature COVERS the 
        # ground truth and count it for the final result
        if ratio > 0.5:
            good_features = good_features + 1
    
    return good_features / len(highlighted_features)
    

def compare_explanations(normal_mask, grid_mask, blocks_coordinates):
    
    matching_blocks = 0
    total_green_blocks = len(blocks_coordinates)
    
    # Iterate through each block of the grid.
    # coordinate = ( (x,y), (x',y') )
    for coordinate in blocks_coordinates:
        
        top_left = coordinate[0]
        bottom_right = coordinate[1]
        
        block_width = bottom_right[1] - top_left[1] + 1
        block_height = bottom_right[0] - top_left[0] + 1
        block_area = block_width * block_height
        
        green_pixels_count = 0
        
        for i,j in itertools.product(range(top_left[0], bottom_right[0]), range(top_left[1], bottom_right[1])):
                
            if(grid_mask[i,j] == 0):
                total_green_blocks -= 1
                break
            
            # We found a green pixel in the same place of both explanations
            if(normal_mask[i,j] == grid_mask[i,j]):
                green_pixels_count += 1
            
        if(green_pixels_count >= (block_area / 2)):
            matching_blocks += 1
            
    
    return float(matching_blocks) / total_green_blocks


#def unique_by_first_n(n, coll):
#    seen = set()
#    for item in coll:
#        compare = tuple(item[:n])    # Keep only the first `n` elements in the set
#        if compare not in seen:
#            seen.add(compare)
#            yield item

#def evaluate_explanations(explainer,
#                          model,
#                          bb_outcomes,
#                          images_to_explain,
#                          ground_truths,
#                          neigh_size=100,
#                          segmentation_fun=None,
#                          hide_col=0,
#                          shown_features=10000): 
#    
#    """
#    Evaluates the quality of the explanations of the given explainer
#    on a set of images
#    
#    Args:
#        explainer:         the explainer whose explanations have to be 
#                           evaluated;
#        model:             the black box model;
#        bb_outcomes:       predictions of the black box;        
#        images_to_explain: images in which we want to evaluate explanations
#        ground_truths:     cut images, used as oracle when evaluating
#                           explanations;
#        neigh_size:        number of images generated in the neighborhood
#                           of each image;
#        segmentation_fun:  defines the shape of the explanation's features;
#        hide_col:          when using basic versions of Lime, this is the
#                           color corresponding to a turned-off feature;
#        shown_features:    number of features shown in the explanation;
#    
#    """
#    
#    # Each entry of this list consist of a pair (rel, abs) where
#    # rel = relative quality of the image's explanation
#    # abs = absolute quality of the image's explanation
#    list_of_qualities = []
#    
#    for i in range(len(images_to_explain)):
#        
#        # Explanation of the i-th image, using the explainer pas-
#        # sed as a parameter
#        explanation = explainer.explain_instance(images_to_explain[i],
#                                                 model.predict,
#                                                 top_labels=1,
#                                                 hide_color=hide_col,
#                                                 num_samples=neigh_size,
#                                                 segmentation_fn=segmentation_fun)
#        
#        # Black box prediction of the i-th image
#        label_to_exp = bb_outcomes[i]
#        
#        # Features highlighted by the explanation. We take the top
#        # 'shown_features' positive features
#        highlighted_features = [area[0] for area in \
#                                explanation.local_exp[label_to_exp] if area[1] > 0][:shown_features]
#        
#        
#        rel_quality = relative_quality_of_explanation(ground_truths[i],
#                                                   highlighted_features,
#                                                   explanation.segments)
#        
#        abs_quality = absolute_quality_of_explanation(ground_truths[i],
#                                                   highlighted_features,
#                                                   explanation.segments)
#        
#        list_of_qualities.append((rel_quality, abs_quality))
#                                                 
#    return np.array(list_of_qualities)
            
            
            
