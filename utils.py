import json
import os
from tkinter import Tk  
from tkinter.filedialog import askopenfilename
import deeplabcut as dlc


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return(data)

def ask_filepath():
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() 
    root.update()
    return(filename)

def load_config():
    config = load_json('lib/config.json')
    config['openfield_config_path'] = os.path.join(config['openfield_root'], 'config.yaml')
    return(config)

def select(hellowords, array, defaultChoose = None, **kwargs):
    # This function helps you use input function to select a key value from an array.
    # It list the elements and use choose the element by idx. Based on the idx, the 
    # function return the value. if you want to add some extra words, use extra_note = 'xxx'.
    if (defaultChoose != None):
        hellowords = hellowords + ' Press Enter for %s' % array[defaultChoose]
    print(hellowords)
    
    for i in range(len(array)):
        print('%d ---> %s' % (i, array[i]))
    x = input('Select by idx: ')
    if x == '':
        if defaultChoose != None:
            final = array[defaultChoose]
        else:
            pass
    else:
        final = array[int(x)]

    try:
        final = final + ' (' + kwargs['extra_note'] + ')'
    except:
        pass

    return(final)

def chooseProject():
    projectlist = os.listdir('.')
    projectlist = [x for x in projectlist if x.split('_')[0] == 'project']
    project = select('Choose the project: ', projectlist, defaultChoose=0)
    return(project)

def startProject():
    # This function is mainly just to avoid ipynb running deeplabcut caused crush. 
    # Using terminal won't have error. 
    config = load_config()
    task = input('Task (also the folder name): ')
    name = input('Project author: ')
    foldername = os.path.join(config['project_root_folder'], task)
    print(foldername)
    
    # we don't want a full GUI, so keep the root window from appearing
    filename = ask_filepath() # show an "Open" dialog box and return the path to the selected file
    print(filename)

    #dlc.create_new_project(task, name, [filename], working_directory=foldername, copy_videos=True, multianimal=False)
    dlc.create_new_project(task, name, [filename], working_directory=foldername, copy_videos=True)

def analyze_video(create_labeled_video = True):
    # This function is to analyze new video chose by popup window.
    # Check deeplabcut github instruction (I) Novel Video Analysis.

    config = load_config()
    filename = ask_filepath()
    
    dlc.analyze_videos(config['openfield_config_path'], [filename], save_as_csv=True)
    if create_labeled_video:
        dlc.create_labeled_video(config['openfield_config_path'], [filename], save_frames = True)

def create_labeled_video(videopath):
    config = load_config()
    dlc.create_labeled_video(config['openfield_config_path'], [videopath], save_frames = True)


