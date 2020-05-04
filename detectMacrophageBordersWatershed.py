import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imutils

# 1. Get image
image_path = "macrophage_testimg.png"
Original_img = np.uint8(mpimg.imread(image_path))
hfigure = plt.figure(1)
plt.imshow(Original_img, cmap='gray')
hfigure.canvas.set_window_title('1.Original_img')

# 2. Convert to BRG format manually, use this function if image is binary.
NewImage = np.uint8(np.empty(shape=(1000, 1000, 3)))
Original_img[Original_img == 1] = 255
for idx in range(3):
    NewImage[:, :, idx] = Original_img

# 3. Pad images with black at 10% of length and width
top = int(0.01*NewImage.shape[0])
bottom = top
left = int(0.01*NewImage.shape[1])
right = left
PaddedImage = cv2.copyMakeBorder(NewImage, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
hfigure = plt.figure(2)
hfigure.canvas.set_window_title('2.Padded img')
plt.imshow(PaddedImage)

# 4.Convert to grayscale
gray = cv2.cvtColor(PaddedImage, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# f, axarr = plt.subplots(3, 3, constrained_layout=True)
# axarr[0, 0].imshow(Original_img, cmap="gray")
# axarr[0, 0].set_title('1.Original_img')

# 5. Dilate to find sure background (Everything = 0 is background)
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=2)
hfigure = plt.figure(3)
hfigure.canvas.set_window_title('3. Sure Background')
plt.imshow(sure_bg, cmap='gray')

# 6. Distance transform to find sure foreground

dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
hfigure = plt.figure(4)
hfigure.canvas.set_window_title('4. Sure Foreground')
plt.imshow(sure_fg, cmap='gray')

# Clean noise
# opening = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
# opening2 = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=2)
# f, axarr = plt.subplots(1, 2, constrained_layout=True)
# axarr[0].imshow(opening, cmap='gray')
# axarr[0].set_title('1')
# axarr[1] = plt.imshow(opening2, cmap='gray')
# axarr[0].set_title('2')

# 7. Find areas that we are unsure that may be borders
sure_fg = np.uint8(sure_fg)
sure_bg = np.uint8(sure_bg)
unknown = cv2.subtract(sure_bg, sure_fg)
hfigure = plt.figure(5)
hfigure.canvas.set_window_title('5. Unknown')
plt.imshow(unknown, cmap='gray')

# 8. Mark unknown regions = 0
#  The regions we know for sure (whether foreground or background)
#  are labelled with any positive integers, but different integers,
#  and the area we don’t know for sure are just left as zero.
ret, markers = cv2.connectedComponents(sure_fg)

# 9. Add one to all labels so that sure background is not 0, but 1
markers1 = markers+1

# 10. Now, mark the region of unknown with zero
markers1[unknown==255] = 0
hfigure = plt.figure(6)
hfigure.canvas.set_window_title('6. markers')
plt.imshow(markers1, cmap='gray')

# 11. Apply watershed to detect boundaries, note watershed function takes Input 8-bit 3-channel image only.
final = cv2.watershed(PaddedImage, markers1)
PaddedImage[final == -1] = [255, 0, 0] # Mark the boundaries (values = -1) found in watershed to red.
hfigure = plt.figure(7)
hfigure.canvas.set_window_title('7. Padded Final image')
plt.imshow(PaddedImage, cmap='gray')

# 12. Crop and remove padding of image, leave 1% of padding
# a = int(0.009*NewImage.shape[0])
# b = int((PaddedImage.shape[0]) - (0.009*NewImage.shape[0]))
# c = int(0.009*NewImage.shape[1])
# d = int((PaddedImage.shape[1]) - (0.009*NewImage.shape[1]))
# crop_img = PaddedImage[a:b, c:d]
# hfigure = plt.figure(8)
# hfigure.canvas.set_window_title('8. Finale image')
# plt.imshow(crop_img, cmap='gray')
#
# # Remove padding
# BordersImg = final[a:b, c:d]

# Create array to append extracted data shape=(1000, 1000, 3))
extracted_data = []

# Create a copy of paddedimage as boundingbox data keep appearing in extracted data.
# copyofPaddedImage = np.empty(shape = (PaddedImage.shape), dtype="uint8")
# copyofPaddedImage[:,:,0] = PaddedImage[:,:,0]
# copyofPaddedImage[:,:,1] = PaddedImage[:,:,1]
# copyofPaddedImage[:,:,2] = PaddedImage[:,:,2]

# Padded_Image = cv2.copyMakeBorder(NewImage, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

# For ref go to https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
for objectIdx in np.unique(final):
    if objectIdx in (1, -1):  # background equals 1
        continue
    # Allocate memory and draw the shape into mask
    mask = np.zeros(shape = (final.shape), dtype="uint8")
    mask[final == objectIdx] = 255
    mask = np.uint8(mask)
    cuntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuntrs = imutils.grab_contours(cuntrs)
    c = max(cuntrs, key=cv2.contourArea)    # contour area calculates the contour area.
    # Documentation for min bounding rectangles available at
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    x, y, w, h = cv2.boundingRect(c)
    ROI = mask[y-5:y + h + 5, x-5:x + w+5]
    extracted_data.append(ROI)
    cv2.rectangle(PaddedImage, (x, y), (x + w, y + h), (255, 255, 255), 1) # weird behaviour here. If this code is executed, ROI changes to also included bounding box even when executed after the data had been appended to list, to solve this you have to create a copy of the array.
    cv2.putText(PaddedImage, "#{}".format(objectIdx-2), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # cv2.drawContours(crop_img, [box.astype(int)], 0, (255, 255, 255), 1)
    # cv2.putText(crop_img, "#{}".format(objectIdx), (int(x) - 10, int(y)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# index = np.where(BordersImg==-1)
# listOfIndices = list(zip(index[0], index[1]))
# for indice in listOfIndices:
#     print(indice)
hfigure = plt.figure(9)
hfigure.canvas.set_window_title('9. Bounding Box image')
plt.imshow(PaddedImage)
plt.show()

# change crop img to something that has not been marked, no red lines
#Extract image to individual boxes and classify them

