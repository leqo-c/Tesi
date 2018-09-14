import numpy as np
import sklearn
import sklearn.preprocessing
from skimage.color import gray2rgb

from lime_image import LimeImageExplainer, ImageExplanation
from .wrappers.scikit_image import SegmentationAlgorithm

class LimeImagePatchworkExplainer(LimeImageExplainer): 

    # Extend LineImageExplainer so that  a new parameter is added. This
    # parameter holds the collection of images to draw the fudged image
    # (i.e. image to show when superpixel is turned off) from.
    def __init__(self, image_pool, *args, **kwargs):
        super(LimeImagePatchworkExplainer, self).__init__(*args, **kwargs)
        self.image_pool = image_pool

    # LimeImagePatchworkExplainer  ovverides the explain_instance method 
    # so that the fudged_image passed to the method data_labels is drawn 
    # at random from the ones inside 'image_pool'.
    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         return_sample_neighborhood_images=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
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

        # fudged_image = image.copy()
        # if hide_color is None:
        #     for x in np.unique(segments):
        #         fudged_image[segments == x] = (
        #             np.mean(image[segments == x][:, 0]),
        #             np.mean(image[segments == x][:, 1]),
        #             np.mean(image[segments == x][:, 2]))
        # else:
        #     fudged_image[:] = hide_color

        # Draw the fudged_image at random
        import random
        fudged_image = random.choice(self.image_pool)

        top = labels

        if(return_sample_neighborhood_images):
            data, labels, samples = self.data_labels(image, fudged_image, segments,
                                                     classifier_fn, num_samples,
                                                     batch_size=batch_size, 
                                                     return_sample_neighborhood_images=return_sample_neighborhood_images,
                                                     fudged_images_pool=self.image_pool)
        else:
            data, labels = self.data_labels(image, fudged_image, segments,
                                            classifier_fn, num_samples,
                                            batch_size=batch_size, 
                                            return_sample_neighborhood_images=return_sample_neighborhood_images,
                                            fudged_images_pool=self.image_pool)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        if(return_sample_neighborhood_images):             
            return ret_exp, samples
        else:
            return ret_exp