class detectMacrophages:

    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    import imutils
    import pickle
    import tkinter as Tk
    from PIL import Image, ImageTk
    from tkinter import messagebox



    def __init__(self, img_file):
        self.img_file = img_file

    def showImages(self, data, figureNumber, title, *argument):

        hfigure = self.plt.figure(figureNumber)
        hfigure.canvas.set_window_title(title)
        if len(argument) == 1:
            self.plt.imshow(data, cmap=argument[0])
        else:
            self.plt.imshow(data)

    def segmentImage(self, showImageData):

        showImageData = bool(showImageData)

        # 1. Get image
        image_path = self.img_file
        Original_img = self.np.uint8(self.mpimg.imread(image_path))

        # 2. Convert to BRG format manually, use this function if image is binary.
        NewImage = self.np.uint8(self.np.empty(shape=(1000, 1000, 3)))
        Original_img[Original_img == 1] = 255
        for idx in range(3):
            NewImage[:, :, idx] = Original_img

        # 3. Pad images with black at 10% of length and width
        top = int(0.01 * NewImage.shape[0])
        bottom = top
        left = int(0.01 * NewImage.shape[1])
        right = left
        PaddedImage = self.cv2.copyMakeBorder(NewImage, top, bottom, left, right, self.cv2.BORDER_CONSTANT, None, 0)

        # 4.Convert to grayscale
        gray = self.cv2.cvtColor(PaddedImage, self.cv2.COLOR_BGR2GRAY)
        ret, thresh = self.cv2.threshold(gray, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)

        # 5. Dilate to find sure background (Everything = 0 is background)
        kernel = self.np.ones((3, 3), self.np.uint8)
        sure_bg = self.cv2.dilate(thresh, kernel, iterations=2)

        # 6. Distance transform to find sure foreground
        dist_transform = self.cv2.distanceTransform(thresh, self.cv2.DIST_L2, 5)
        ret, sure_fg = self.cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)

        # 7. Find areas that we are unsure that may be borders
        sure_fg = self.np.uint8(sure_fg)
        sure_bg = self.np.uint8(sure_bg)
        unknown = self.cv2.subtract(sure_bg, sure_fg)

        # 8. Mark unknown regions = 0
        #  The regions we know for sure (whether foreground or background)
        #  are labelled with any positive integers, but different integers,
        #  and the area we don’t know for sure are just left as zero.
        ret, markers = self.cv2.connectedComponents(sure_fg)

        # 9. Add one to all labels so that sure background is not 0, but 1
        markers1 = markers + 1

        # 10. Now, mark the region of unknown with zero
        markers1[unknown == 255] = 0

        # 11. Apply watershed to detect boundaries, note watershed function takes Input 8-bit 3-channel image only.
        final = self.cv2.watershed(PaddedImage, markers1)
        PaddedImage[final == -1] = [255, 0, 0]
        # Mark the boundaries (values = -1) found in watershed to red.

        # 12. Create array to append extracted data.
        extracted_data = []

        # 13. Draw Bounding box and extra each macrophage into single array.
        # For ref go to https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
        for objectIdx in self.np.unique(final):
            if objectIdx in (1, -1):  # background equals 1
                continue
            # Allocate memory and draw the shape into mask
            mask = self.np.zeros(shape=(final.shape), dtype="uint8")
            mask[final == objectIdx] = 255
            mask = self.np.uint8(mask)
            cuntrs = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
            cuntrs = self.imutils.grab_contours(cuntrs)
            c = max(cuntrs, key=self.cv2.contourArea)
            # contour area calculates the contour area.
            # Documentation for min bounding rectangles available at
            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            x, y, w, h = self.cv2.boundingRect(c)
            ROI = mask[y - 5:y + h + 5, x - 5:x + w + 5]
            extracted_data.append(ROI)

            # weird behaviour here. If this code is executed, ROI changes to also included
            # bounding box even when executed after the data had been appended to list,
            # to solve this you have to create a copy of the array.

            self.cv2.rectangle(PaddedImage, (x, y), (x + w, y + h), (255, 255, 255),
                          1)

            self.cv2.putText(PaddedImage, "#{}".format(objectIdx - 2), (int(x) - 10, int(y)),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        if showImageData:
            self.showImages(Original_img, 1, '1. Original_img', 'gray')
            self.showImages(sure_bg, 2, '2.Sure Background', 'gray')
            self.showImages(sure_fg, 3, '3.Sure Foreground', 'gray')
            self.showImages(unknown, 4, '4.Unknown', 'gray')
            self.showImages(markers1, 5, '5.Markers + 1', 'gray')
            self.showImages(PaddedImage, 6, '6.SegmentedImage', 'gray')
            self.plt.show()

        return extracted_data

    def manual_classification(self, extracted_data):

        logical_list = self.np.ones(shape=(200, 1), dtype=bool)
        image_number = 0

        def Ramified_Macrophage():

            nonlocal logical_list
            nonlocal image_number
            logical_list[image_number] = True
            image_number += 1
            arrayImage = self.Image.fromarray(extracted_data[image_number])
            resizedImage = arrayImage.resize((200, 200))
            img = self.ImageTk.PhotoImage(image=resizedImage)
            canvas.create_image(0, 0, anchor=self.Tk.NW, image=img)
            root.mainloop()

        def Amoeboid_Macrophage():

            nonlocal logical_list
            nonlocal image_number
            image_number
            logical_list
            logical_list[image_number] = False
            image_number += 1
            arrayImage = self.Image.fromarray(extracted_data[image_number])
            resizedImage = arrayImage.resize((200, 200))
            img = self.ImageTk.PhotoImage(image=resizedImage)
            canvas.create_image(0, 0, anchor=self.Tk.NW, image=img)
            root.mainloop()

        def on_closing():
            if self.messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()


        root = self.Tk.Tk()
        arrayImage = self.Image.fromarray(extracted_data[image_number])
        resizedImage = arrayImage.resize((200, 200))
        img = self.ImageTk.PhotoImage(image=resizedImage)

        canvas = self.Tk.Canvas(root, width=200, height=200)
        canvas.create_image(0, 0, anchor=self.Tk.NW, image=img)

        fr_buttons = self.Tk.Frame(root)
        btn_ramified = self.Tk.Button(fr_buttons, text="Ramified", command=Ramified_Macrophage)
        btn_amoeboid = self.Tk.Button(fr_buttons, text="Amoeboid", command=Amoeboid_Macrophage)

        canvas.grid(row=0, column=0, rowspan=4, columnspan=3)
        fr_buttons.grid(row=4, column=1, sticky="ns")
        btn_ramified.grid(sticky="ew", padx=5, pady=5)
        btn_amoeboid.grid(sticky="ew", padx=5, pady=5)
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        return logical_list
