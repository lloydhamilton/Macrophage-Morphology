from ID_Macrophages import class_detectMacrophages as Dm
from matplotlib import pyplot as plt
import pickle
import tkinter as Tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import messagebox
# classInstance = Dm.detectMacrophages('testImage2.png')
# segmented = classInstance.segmentImage(0)

# Save data variable
# with open('SegmentedImages', 'wb') as f:
#     pickle.dump(segmented, f)

# import tkinter as tk
# window = tk.Tk()
# # window.mainloop()
# root.mainloop()
# Load data variable

with open('SegmentedImages', 'rb') as f:
    segmentedImages = pickle.load(f)
logical_list = np.ones(shape=(200, 1), dtype=bool)
image_number = 0

def Ramified_Macrophage():
    global image_number
    global logical_list
    logical_list[image_number] = True
    image_number += 1
    arrayImage = Image.fromarray(segmentedImages[image_number])
    resizedImage = arrayImage.resize((200, 200))
    img = ImageTk.PhotoImage(image=resizedImage)
    canvas.create_image(0, 0, anchor=Tk.NW, image=img)
    root.mainloop()

def Amoeboid_Macrophage():
    global image_number
    global logical_list
    logical_list[image_number] = False
    image_number += 1
    arrayImage = Image.fromarray(segmentedImages[image_number])
    resizedImage = arrayImage.resize((200, 200))
    img = ImageTk.PhotoImage(image=resizedImage)
    canvas.create_image(0, 0, anchor=Tk.NW, image=img)
    root.mainloop()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        print(logical_list)
        return logical_list


root = Tk.Tk()
arrayImage = Image.fromarray(segmentedImages[image_number])
resizedImage = arrayImage.resize((200, 200))
img = ImageTk.PhotoImage(image=resizedImage)

canvas = Tk.Canvas(root, width=200, height=200)
canvas.create_image(0, 0, anchor=Tk.NW, image=img)

fr_buttons = Tk.Frame(root)
btn_ramified = Tk.Button(fr_buttons, text="Ramified", command=Ramified_Macrophage)
btn_amoeboid = Tk.Button(fr_buttons, text="Amoeboid", command=Amoeboid_Macrophage)

canvas.grid(row=0, column=0, rowspan=4, columnspan=3)
fr_buttons.grid(row=4, column=1, sticky="ns")
btn_ramified.grid(sticky="ew", padx=5, pady=5)
btn_amoeboid.grid(sticky="ew", padx=5, pady=5)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

# frame = Tk.Frame(master=root, relief=Tk.RAISED, borderwidth=5)
# frame.pack(side=Tk.LEFT)
# label = Tk.Label(master=frame, text='relief_name')
# label.pack()

# plt.figure()
# classInstance = Dm.detectMacrophages('testImage2.png')
# classInstance.showImages(segmentedImages[2], 1, 'Data1', 'gray')
# plt.show()
