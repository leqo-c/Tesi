import sys
sys.path.append('/home/leo/Desktop/Tesi/tf-models/slim')
from datasets import imagenet

names = imagenet.create_readable_names_for_imagenet_labels()


# PARAM 
#    predictions: a list of tuples (u'n02114548', u'white_wolf', 0.89647627)
#
# RETURNS: a list of triples of the form
#          (u'airliner', 403, 0.9781376)
def get_imagenet_labels(predictions):
    res = []
    for p in predictions:
        
        class_name = p[1].replace("_", " ")
        
        label = [ k-1 for k in names.keys() if class_name == names[k].split(",")[0] ]
        res.append((p[1], label[0], p[2]))
    return res

