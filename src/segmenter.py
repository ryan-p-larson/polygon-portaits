from typing import List, Optional
import numpy as np
import cv2
from preprocessor import PreProcessor

class Segmenter(object):
  def __init__(self, model=None):
    self.model = model

  @classmethod
  def mask(cls,
    image: np.ndarray,
    segmented: np.ndarray,
    include: Optional[List[int]] = [],
    exclude: Optional[List[int]] = [0]) -> (np.ndarray, np.ndarray):

    # first create mask from segments matching our criteria
    include, exclude = set(include), set(exclude)
    mask             = np.zeros(image.shape[:2], dtype=np.uint8)
    print(f"include={include},\texclude={exclude}")

    for row in range(image.shape[0]):
      for col in range(image.shape[1]):
        cls            = segmented[row][col]
        included       = (cls in include) if (len(include) > 0) else True
        excluded       = cls in exclude
        mask[row][col] = 1 if (included and not excluded) else 0

    # apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=cv2.UMat(mask))

    return cv2.UMat.get(masked_image), mask

  def segment(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
    # First, resize the image if it's too small
    height_og, width_og = image.shape[:2]
    resized_image = PreProcessor.scale_down(image, 1200)

    # Second, segment images according to model
    faces = self.model.parse_face(resized_image)
    segs  = faces[0]

    # Construct mask by filtering/normalizing classes
    masked, mask = self.mask(resized_image, segs, exclude=[0, 16])

    # Finally, resize the image to the original image passed in
    resized_mask = PreProcessor.scale_up(mask, max(height_og, width_og))
    resized_segs = PreProcessor.scale_up(segs, max(height_og, width_og))

    return resized_mask, resized_segs