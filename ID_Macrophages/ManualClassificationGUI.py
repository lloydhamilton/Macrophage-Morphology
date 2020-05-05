from ID_Macrophages import class_detectMacrophages as Dm
from matplotlib import pyplot as plt
import pickle
import tkinter as Tk
from PIL import Image, ImageTk

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

with open ('SegmentedImages', 'rb') as f:
    segmentedImages = pickle.load(f)

root = Tk.Tk()
arrayImage = Image.fromarray(segmentedImages[2])
resizedImage = arrayImage.resize((200,200))
img = ImageTk.PhotoImage(image=resizedImage)

canvas = Tk.Canvas(root, width = 200, height = 200)
canvas.create_image(0,0, anchor=Tk.NW, image=img)

fr_buttons = Tk.Frame(root)
btn_open = Tk.Button(fr_buttons, text="Open")
btn_save = Tk.Button(fr_buttons, text="Save As...")

canvas.grid(row=0, column=0, columnspan=3, rowspan=3)
fr_buttons.grid(row=4, column=1, sticky="ns")
btn_open.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=2, column=0, sticky="ew", padx=5)
Tk.mainloop()

# frame = Tk.Frame(master=root, relief=Tk.RAISED, borderwidth=5)
# frame.pack(side=Tk.LEFT)
# label = Tk.Label(master=frame, text='relief_name')
# label.pack()

plt.figure()
classInstance = Dm.detectMacrophages('testImage2.png')
classInstance.showImages(segmentedImages[2], 1, 'Data1', 'gray')
plt.show()

