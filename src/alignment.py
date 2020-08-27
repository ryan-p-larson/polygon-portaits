
from os.path import abspath, dirname, join, exists
import numpy as np
import cv2
import dlib
from imutils.face_utils import FACIAL_LANDMARKS_IDXS, shape_to_np

DEFAULT_LANDMARKS_PATH = join(abspath(dirname(__file__)), "imgs", "shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor(DEFAULT_LANDMARKS_PATH)
one_path  = join(abspath(dirname(__file__)), "imgs", "before_align.jpg")
two_path  = join(abspath(dirname(__file__)), "imgs", "after_align.jpg")
one       = cv2.imread(one_path)


class Alignment:
  def __init__(self,
    landmarks_path: str = DEFAULT_LANDMARKS_PATH):
    self.detector  = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(landmarks_path)

  def distance(self, a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

  def cosine_formula(self, length_line1, length_line2, length_line3):
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

  def is_between(self, point1, point2, point3, extra_point) -> bool:
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    return (((c1 < 0) & (c2 < 0) & (c3 < 0)) | ((c1 > 0) & (c2 > 0) & (c3 > 0)))

  def align(self, image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = self.detector(gray, 1)

    for rect in detections:
      x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
      shape = self.predictor(gray, rect)
      shape_arr = shape_to_np(shape)

      nose = shape_arr[27:36].copy().mean(axis=0).astype('int32')
      left_eye = shape_arr[42:48].copy().mean(axis=0).astype('int32')
      right_eye = shape_arr[36:42].copy().mean(axis=0).astype('int32')

      center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
      center_pred = (int((x + w) / 2), int((y + y) / 2))

      line1 = self.distance(center_of_forehead, nose)
      line2 = self.distance(center_pred, nose)
      line3 = self.distance(center_pred, center_of_forehead)
      cos_a = self.cosine_formula(line1, line2, line3)
      angle = np.arccos(cos_a)

      rotated_point = self.rotate_point(nose, center_of_forehead, angle)
      rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
      if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
        angle = np.degrees(-angle)
      else:
        angle = np.degrees(angle)
      # print(center_of_forehead, center_pred, cos_a, angle, rotated_point, angle)

      return self.rotate_image(image, nose, angle)

def rotation_detection_dlib(img, mode, show=False):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)
  if len(rects) > 0:
    for rect in rects:
      x = rect.left()
      y = rect.top()
      w = rect.right()
      h = rect.bottom()

      shape = predictor(gray, rect)
      shape = shape_to_normal(shape)

      nose, left_eye, right_eye = get_eyes_nose_dlib(shape)

      center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
      center_pred = (int((x + w) / 2), int((y + y) / 2))

      length_line1 = distance(center_of_forehead, nose)
      length_line2 = distance(center_pred, nose)
      length_line3 = distance(center_pred, center_of_forehead)

      cos_a = cosine_formula(length_line1, length_line2, length_line3)

      angle = np.arccos(cos_a)
      rotated_point = rotate_point(nose, center_of_forehead, angle)
      rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

      if is_between(nose, center_of_forehead, center_pred, rotated_point):
        angle = np.degrees(-angle)
      else:
        angle = np.degrees(angle)

      if mode:
        img = rotate_opencv(img, nose, angle)
      else:
        img = np.array(img.rotate(angle))
    if show:
      show_img(img)
    return img
  else:
    return img






alignment = Alignment(DEFAULT_LANDMARKS_PATH)
test_one = alignment.align(one)
cv2.imshow('Test alignmnet', test_one)
while True:
  ch = cv2.waitKey(1)
  if ch == 27 or ch == ord('q') or ch == ord('Q'):
    break
cv2.destroyAllWindows()