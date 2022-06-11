import cv2
import numpy as np
import matplotlib.pyplot as plt

nemo = cv2.imread('nemo.jpg')
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result = cv2.bitwise_and(nemo, nemo, mask=mask)

plt.subplot(1, 4, 1)
plt.imshow(nemo)
plt.title('Citra Original')
plt.subplot(1, 4, 2)
plt.imshow(hsv_nemo)
plt.title('Citra HSV')
plt.subplot(1, 4, 3)
plt.imshow(mask)
plt.title('Citra Thresholding')
plt.subplot(1, 4, 4)
plt.imshow(result)
plt.title('Citra Hasil')
plt.show()