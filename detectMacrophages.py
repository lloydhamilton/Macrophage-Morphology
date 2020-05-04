class detectMacrophages:

    def __init__(self, img_file, showImageData):
        self.img_file = img_file
        self.showImageData = showImageData

    def showImages(self, data, figureNumber, title, *argument):

        from matplotlib import pyplot as plt

        hfigure = plt.figure(figureNumber)
        hfigure.canvas.set_window_title(title)
        if len(argument) == 1:
            plt.imshow(data, cmap=argument)
        else:
            plt.imshow(data)

    def segmentImage (self):

        import numpy as np
        import cv2
        from matplotlib import pyplot as plt
        import matplotlib.image as mpimg
        import imutils

        # 1. Get image
        image_path = self.img_file
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
        top = int(0.01 * NewImage.shape[0])
        bottom = top
        left = int(0.01 * NewImage.shape[1])
        right = left
        PaddedImage = cv2.copyMakeBorder(NewImage, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

        # 4.Convert to grayscale
        gray = cv2.cvtColor(PaddedImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Dilate to find sure background (Everything = 0 is background)
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=2)

        # 6. Distance transform to find sure foreground
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)

        # 7. Find areas that we are unsure that may be borders
        sure_fg = np.uint8(sure_fg)
        sure_bg = np.uint8(sure_bg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 8. Mark unknown regions = 0
        #  The regions we know for sure (whether foreground or background)
        #  are labelled with any positive integers, but different integers,
        #  and the area we don’t know for sure are just left as zero.
        ret, markers = cv2.connectedComponents(sure_fg)

        # 9. Add one to all labels so that sure background is not 0, but 1
        markers1 = markers + 1

        # 10. Now, mark the region of unknown with zero
        markers1[unknown == 255] = 0

        # 11. Apply watershed to detect boundaries, note watershed function takes Input 8-bit 3-channel image only.
        final = cv2.watershed(PaddedImage, markers1)
        PaddedImage[final == -1] = [255, 0, 0]
        # Mark the boundaries (values = -1) found in watershed to red.

        # 12. Create array to append extracted data.
        extracted_data = []

        # 13. Draw Bounding box and extra each macrophage into single array.
        # For ref go to https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
        for objectIdx in np.unique(final):
            if objectIdx in (1, -1):  # background equals 1
                continue
            # Allocate memory and draw the shape into mask
            mask = np.zeros(shape=(final.shape), dtype="uint8")
            mask[final == objectIdx] = 255
            mask = np.uint8(mask)
            cuntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cuntrs = imutils.grab_contours(cuntrs)
            c = max(cuntrs, key=cv2.contourArea)
            # contour area calculates the contour area.
            # Documentation for min bounding rectangles available at
            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            x, y, w, h = cv2.boundingRect(c)
            ROI = mask[y - 5:y + h + 5, x - 5:x + w + 5]
            extracted_data.append(ROI)

            # weird behaviour here. If this code is executed, ROI changes to also included
            # bounding box even when executed after the data had been appended to list,
            # to solve this you have to create a copy of the array.

            cv2.rectangle(PaddedImage, (x, y), (x + w, y + h), (255, 255, 255),
                          1)

            cv2.putText(PaddedImage, "#{}".format(objectIdx - 2), (int(x) - 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            if self.showImageData == 1:

                continue