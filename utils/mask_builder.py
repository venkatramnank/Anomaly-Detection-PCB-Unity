"""
Mask builder 
Builds binary masks from segmentation masks using the info from Unity json files
Needs to be manually done for each set of segmentation mask and each json file
TODO: NEED TO AUTOMATE
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def segmentation_mask_convertor_nail(img_path):
    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    #TODO(Venkat):Need to automatically fetch these values
    """
    {
              "labelName": "  52",
              "pixelValue": [
                94, # R
                95, # G
                255, # B
                255
              ]
            }
    
    """
    rgb_val =  (255, 95, 94) # Needs to be flipped since cv2 reads as BGR instead of RGB
    rgb_val2 = (255, 213, 38)
    lower_range = np.array(rgb_val, dtype=np.uint8)
    upper_range = np.array(rgb_val, dtype=np.uint8)
    lower_range2 = np.array(rgb_val2, dtype=np.uint8)
    upper_range2 = np.array(rgb_val2, dtype=np.uint8)
    mask = cv2.inRange(image, lower_range, upper_range)
    mask2 = cv2.inRange(image, lower_range2, upper_range2)
    combined_mask = cv2.bitwise_or(mask, mask2)
    result = np.zeros_like(image)
    # result[combined_mask == 255] = [255,255,255] # Uncomment if combined mask to be used
    result[mask == 255] = [255, 255, 255] # Comment if above line uncommented
    cv2.imwrite(r'C:\Users\kalyanak\Desktop\FiabFlexforce\AnomalyDetection\nokia_pcb_dataset\OneDrive_2023-06-29\PCB DATA\utils\solo_43.png', result)

test_img_path = r'C:\Users\kalyanak\Desktop\FiabFlexforce\AnomalyDetection\nokia_pcb_dataset\OneDrive_2023-06-29\PCB DATA\PCB-ErrorData\solo_43\sequence.0\step0.camera.semantic segmentation.png'
segmentation_mask_convertor_nail(test_img_path)