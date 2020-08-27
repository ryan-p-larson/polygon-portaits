# Polygon Portraits


#### Steps: High level

1. Read in two photos (henceforth `before`/`after`)
2. Align faces using landmarks, transform accordingly and bring to a standard `1080*1080`
3. Assign parameters to facial transform (eg whitespace margin). Clip `before/after`
4. run segmentation on the outline of head/shoulder, and make background transparent
  - base layer is white
  - shadow layer includes some gaussian blur of transparent black
5. Alter foreground color intensities => histogram equalization
6.