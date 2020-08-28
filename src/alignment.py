
from os.path import abspath, dirname, join, exists
import numpy as np
import cv2
import dlib
from imutils.face_utils import FACIAL_LANDMARKS_IDXS, shape_to_np

DEFAULT_LANDMARKS_PATH = join(abspath(dirname(__file__)), "imgs", "shape_predictor_68_face_landmarks.dat")

class Alignment:
  def __init__(self,
    landmarks_path: str = DEFAULT_LANDMARKS_PATH):
    self.detector  = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(landmarks_path)

  def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

  def _is_between(self, pt1, pt2, pt3, extra_pt) -> bool:
    c1 = (pt2[0] - pt1[0]) * (extra_pt[1] - pt1[1]) - (pt2[1] - pt1[1]) * (extra_pt[0] - pt1[0])
    c2 = (pt3[0] - pt2[0]) * (extra_pt[1] - pt2[1]) - (pt3[1] - pt2[1]) * (extra_pt[0] - pt2[0])
    c3 = (pt1[0] - pt3[0]) * (extra_pt[1] - pt3[1]) - (pt1[1] - pt3[1]) * (extra_pt[0] - pt3[0])
    return (((c1 < 0) & (c2 < 0) & (c3 < 0)) | ((c1 > 0) & (c2 > 0) & (c3 > 0)))

  def _cosine_formula(self, length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a

  def rotate_image(self, image: np.ndarray, nose_center: np.ndarray, angle: int) -> np.ndarray:
    M       = cv2.getRotationMatrix2D(tuple(nose_center.tolist()), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return rotated

  def rotate_point(self, origin: np.ndarray, point: np.ndarray, angle: int) -> (float, float):
    ox, oy = origin[0], origin[1]
    px, py = point
    qx     = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy     = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

  def detect_face(self, image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = self.detector(gray, 1)

    if (len(detections) == 0):
      return None, None
    else:
      rect = detections[0]
      shape = self.predictor(gray, rect)
      shape_arr = shape_to_np(shape)
      return rect, shape_arr

  def align_face(self, image: np.ndarray):
    # http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/
    rect, shape_arr = self.detect_face(image)
    x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
    nose       = shape_arr[27:36].copy().mean(axis=0).astype('int32')
    left_eye   = shape_arr[42:48].copy().mean(axis=0).astype('int32')
    right_eye  = shape_arr[36:42].copy().mean(axis=0).astype('int32')

    center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    center_pred = (int((x + w) / 2), int((y + y) / 2))

    line1 = self._distance(center_of_forehead, nose)
    line2 = self._distance(center_pred, nose)
    line3 = self._distance(center_pred, center_of_forehead)
    cos_a = self._cosine_formula(line1, line2, line3)
    angle = np.arccos(cos_a)

    rotated_point = self.rotate_point(nose, center_of_forehead, angle)
    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
    if self._is_between(nose, center_of_forehead, center_pred, rotated_point):
      angle = np.degrees(-angle)
    else:
      angle = np.degrees(angle)

    eye_delta = self._distance(left_eye, right_eye)
    # return self.rotate_image(image, nose, angle), eye_delta
    return image, eye_delta

  def scale_faces(self, first: np.ndarray, second: np.ndarray):
    # http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/
    # doesnt assume both images have been aligned first.
    first_align, first_delta   = self.align_face(first)
    second_align, second_delta = self.align_face(second)

    # first image will be used as a reference
    resize_ratio = first_delta / second_delta
    print(f"d1={first_delta}, d2={second_delta}, r={resize_ratio}")

    height_og, width_og = second_align.shape[:2]
    resize_dim = (int(height_og * resize_ratio), int(width_og * resize_ratio))
    resize_second = cv2.resize(second_align, resize_dim)

    comparison = np.hstack([first_align, resize_second])
    return comparison




predictor = dlib.shape_predictor(DEFAULT_LANDMARKS_PATH)
one_path  = join(abspath(dirname(__file__)), "imgs", "before_align.jpg")
two_path  = join(abspath(dirname(__file__)), "imgs", "after_align.jpg")
one       = cv2.imread(one_path)
two = cv2.imread(two_path)

alignment = Alignment(DEFAULT_LANDMARKS_PATH)
# test_one, test_delta = alignment.align_face(one)
test_one = alignment.scale_faces(one, two)
cv2.imshow('Test alignmnet', test_one)
while True:
  ch = cv2.waitKey(1)
  if ch == 27 or ch == ord('q') or ch == ord('Q'):
    break
cv2.destroyAllWindows()