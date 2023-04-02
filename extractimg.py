import cv2
import os
from tkinter import Tk  
from tkinter.filedialog import askopenfilename



def ask_filepath():
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() 
    root.update()
    return(filename)

filepath = ask_filepath()

vidcap = cv2.VideoCapture(filepath)
success,image = vidcap.read()

cv2.imwrite(os.path.join(os.path.dirname(filepath), "sample.jpg"), image)