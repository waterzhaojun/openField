import exp
import os
import shutil
from tkinter import Tk  
from tkinter.filedialog import askopenfilename
import json

# importing the module
import cv2
   
# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.cv.circle(img, x, y, 5, 'red', 5)
        cv2.imshow('image', img)

  
def choose_corners(imagepath):
    img = cv2.imread(imagepath, 1)
  
    # displaying the image
    cv2.imshow('image', img)
  
    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
  
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
  
    # close the window
    cv2.destroyAllWindows()

def ask_filepath():
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() 
    root.update()
    return(filename)

filepath = ask_filepath()
project_setting_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath).split('.')[0].split('DLC')[0] + '_project_setting.json')

if not os.path.exists(project_setting_path):
    shutil.copyfile('lib/project_setting.json', project_setting_path)
    input('Creating %s. Please fill the information in it... \nfps is your video writing frequency. \nCheck corner coordinates in imageJ, replace y,x by the value like [[31, 114], [26, 478], [391, 115], [390, 478]]\ncage_side_length is the length of your cage. Unit is meter. \ntimecourse_length is the video length you want to analyze. It counts from beginning. Unit is second. \When you saved, press any key to continue...' % os.path.basename(project_setting_path))

with open(project_setting_path) as f:
    sett = json.load(f)
    
# The corners' coordinates. You could find it by imageJ. The format should be [y, x] for each corner.
boxcoord = sett['corner_coordinates']

# The cage side length. The unit is meter.
edge_length = sett['cage_side_length']

# The time course length. The unit is second. Usually the total recorded time is more than the required analysis period.
timecourse_length = sett['timecourse_length']

# The video scan rate.
fps = sett['fps']

openf = exp.Openfield(filepath, fps = fps, timecourse_length = timecourse_length)
openf.set_box_info(boxcoord, edge_length)
data = openf.extract_bodypart_data('body')
data_trans = openf.translate_object_coords(data)

# data_relative is the percentage value of the object coordinates. It is the value we will use in following analysis.
data_relative = openf.translate_object_coords_to_percentage(data_trans)

result = openf.analysis_report(data_relative)
print(result)
