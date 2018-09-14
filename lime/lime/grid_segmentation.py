import numpy as np
import math

def old_gridSegmentation(parts, img):
    
    segments = np.zeros(img.shape[:2])
    segments_per_row = img.shape[:2][0] / parts
    
    i_segment_number = 1
    for i in range(0, img.shape[:2][0]):
        
        j_segment_number = i_segment_number - 1
        
        for j in range(0, img.shape[:2][1]):
            
            if (j % segments_per_row) == 0:
                j_segment_number = j_segment_number + 1
                
            segments[i,j] = j_segment_number
            
        if ((i+1) % segments_per_row) == 0:
            i_segment_number = j_segment_number + 1
        
    return segments.astype(int)

def gridSegmentation(parts, img):
    
    segments = np.zeros(img.shape[:2])
    segments_per_row = img.shape[:2][0] / parts
    #segments_per_row = math.ceil(float(img.shape[:2][0]) / parts)
    width = img.shape[:2][0]
    height = img.shape[:2][1]
    
    i_segment_number = 1
    for i in range(0, height):
        
        j_segment_number = i_segment_number - 1
        
        for j in range(0, width):
            
            if ((j % segments_per_row) == 0) and ((j + segments_per_row - 1) < width):
                j_segment_number = j_segment_number + 1
                
            segments[i,j] = j_segment_number
            
        if (((i+1) % segments_per_row) == 0) and ( ((i+1) + segments_per_row -1) < height ):
            i_segment_number = j_segment_number + 1
        
    return segments.astype(int)

def get_blocks_coordinates(segments):
    
    height = segments.shape[0]
    width = segments.shape[1]
    top_left_points = []
    bottom_right_points = []
    
    for i in range(width):
        for j in range(height):
            if ( ((i-1) < 0) or (segments[i-1,j] != segments[i,j]) ) and ( ((j-1) < 0) or (segments[i,j-1] != segments[i,j]) ):
                top_left_points.append((i,j))
            if ( ((i+1) >= width) or (segments[i+1,j] != segments[i,j]) ) and ( ((j+1) >= height) or (segments[i,j+1] != segments[i,j]) ):
                bottom_right_points.append((i,j))
    
    return zip(top_left_points, bottom_right_points)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    