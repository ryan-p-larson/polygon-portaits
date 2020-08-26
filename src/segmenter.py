import numpy as np
from .preprocessor import PreProcessor

class Segmenter(object):
  def __init__(self, model=face_parser.FaceParser):
    self.model = model

  def segment(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
    # First, resize the image if it's too small
    height_og, width_og = image.shape[:2]
    resized_image = PreProcessor.scale_down(image, 1200)

    # Second, segment images according to model
    faces = self.model.parse_face(resized_image)
    segs  = faces[0]

    # Construct mask by filtering/normalizing classes
    mask  = np.zeros(resized_image.shape[:2], dtype=np.uint8)

    for row in range(mask.shape[0]):
      for col in range(mask.shape[1]):
        cls = segs[row][col]
        mask[row][col] = 255 if ((cls > 0) and (cls != 16)) else 0

    # Finally, resize the image to the original image passed in
    resized_mask = PreProcessor.scale_up(mask, max(height_og, width_og))
    resized_segs = PreProcessor.scale_up(segs, max(height_og, width_og))

    return resized_mask, resized_segs
