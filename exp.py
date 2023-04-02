import pandas as pd
import numpy as np
import math
import os
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re
import copy

def angle_translate(coord, angle):
    # The coord is the new coord [y,x].
    # the angle is the value that trun axis clockwise. this value is not degree but real value.
    # This is revised from an old version which the format is [x, y]. Flip many time is really annoy. Need to change to an easier way when have time.
    coord = np.flip(coord)
    res = [
        coord[0] * math.cos(angle) - coord[1] * math.sin(angle),
        coord[0] * math.sin(angle) + coord[1] * math.cos(angle)
    ]
    res = np.flip(res)
    return(res)

def datalength_to_timecourse_label(datalength, fps, gap):
    # This function is to translate the length of data array to time course label.
    # fps is the data is fps. Unit is hz.
    # gap is the gap you want to show a label. Unit is sec.
    if gap < 60:
        labelformat = 'sec'
        labelunit = 1
    elif (gap >= 60) & (gap < 3600):
        labelformat = 'min'
        labelunit = 60
    elif gap >= 3600:
        labelformat = 'h'
        labelunit = 3600

    maxlabel = math.floor(datalength/fps/gap)
    ticks = [x for x in np.arange(0, maxlabel * fps * gap + 1) if x % (gap*fps) == 0]
    labels = [str(int(x/fps/labelunit)) for x in ticks]

    # ticks is tricky. Need to change a bit.
    ticks = np.array(ticks) -1
    ticks[ticks<0] = 0
    ticks = list(ticks.astype(int))

    uniquelabels = np.unique(np.array(labels))
    if len(labels) != len(uniquelabels):
        uniquelabelidx = []
        for u in uniquelabels:
            uniquelabelidx.append(np.where(np.array(labels) == u)[0][0])

        ticks = np.array(ticks)[uniquelabelidx]
        labels = np.array(labels)[uniquelabelidx]
    return(ticks, labels)



class Openfield():
    # Open field experiment. 
    # For all coordinates, format is [y,x], means [r,c]. 
    # xx_raw means the real coordinates in the image file.
    # xx means translated coordinates based on the angle and center.
    # xx_relative means translated from xx to percentage based on the edge length, or cage area.

    def __init__(self, labelpath, fps, timecourse_length, outputfolder = None):
        # timecourse_length: setup how long your experiment is. Usually The video is longer than the required time course. So this parameter will help to crop the data. Unit is sec.
        self.labelpath = labelpath
        self.fps = fps # movie scanrate. Unit is hz.
        self.timecourse_length = timecourse_length 

        if outputfolder == None:
            videoname = os.path.basename(labelpath).split('.')[0].split('DLC')[0]
            self.analysis_report_folder = os.path.join(os.path.dirname(labelpath), videoname + '_analysis_report')
        else:
            self.analysis_report_folder = outputfolder
        
        if not os.path.exists(self.analysis_report_folder):
            os.mkdir(self.analysis_report_folder)
        

    def extract_bodypart_data(self, partname):
        # To extract multiple parts, just run this function several times then combine. I can add another smarter way when have time.
        if self.labelpath.split('.')[-1] == 'csv':
            df = pd.read_csv(self.labelpath, header = [0,1,2]) # I need to add a function to read h5.
            subdf = df.iloc[:, [partname in x for x in df.columns]]
            subdf.columns = [x[2] for x in subdf.columns]
            coords = subdf.loc[:,['y','x']].values # coord's format is [y,x]
        return(coords)

    def set_box_info(self, box_coords, edge_length):
        # box_coords is box's 4 corners' coordinates. The format is [[y,x],[y,x],[y,x],[y,x]], in another words, it is [r,c].
        # edge_length is box's side length. The unit is meter.

        def __sortCorner__(corners):
            # This function is to sort the input corners to let it start from upper left corner and go clockwise.
            center = np.mean(corners, axis = 0)
            c = corners - center
            cup = c[(c[:,0] <= 0)]
            cup_angle = np.arcsin(cup[:,1] / np.sqrt(cup[:,0]**2 + cup[:,1]**2))
            cup_angle_idx = np.argsort(cup_angle)

            cdown = c[(c[:,0] > 0)]
            cdown_angle = np.arcsin(cdown[:,1] / np.sqrt(cdown[:,0]**2 + cdown[:,1]**2))
            cdown_angle_idx = np.flip(np.argsort(cdown_angle))

            res = np.concatenate([cup[cup_angle_idx], cdown[cdown_angle_idx]])

            return(res, center)


        self.box = {
            'edge_length': edge_length
        }

        self.box['corners_raw'] = box_coords
        self.box['corners'],self.box['center_raw'] = __sortCorner__(box_coords) #自然center translate之后就是[0,0]

        point1 = np.mean(self.box['corners'][0:2, :], axis = 0)
        point2 = np.mean(self.box['corners'][2:4, :], axis = 0)
        d = point1 - point2

        point1_backup = np.mean(self.box['corners'][1:3, :], axis = 0)
        point2_backup = np.mean(self.box['corners'][[0,3], :], axis = 0)
        d_backup = point1_backup - point2_backup

        
        if d[0]**2 + d[1]**2 < d_backup[0]**2 + d_backup[1]**2:
            theangle = math.asin(d[1] / math.sqrt(d[0]**2 + d[1]**2))
        else:
            theangle = math.acos(d[1] / math.sqrt(d[0]**2 + d[1]**2))
        
        self.box['angle'] = theangle
        self.box['angle_degree'] = theangle/math.pi*180
        for i in range(np.shape(self.box['corners'])[0]):
            self.box['corners'][i,:] = angle_translate(self.box['corners'][i,:], self.box['angle'])
        
    def translate_object_coords(self, obj_data):
        # Different to percentage translate, This function only move the coordinates based on center and angle.
        data = np.zeros_like(obj_data)
        for i in range(np.shape(obj_data)[0]):
            data[i,:] = angle_translate(obj_data[i,:] - self.box['center_raw'], self.box['angle'])
        return(data)

    def translate_object_coords_to_percentage(self, obj_data):
        # The obj_data should already been rotated and centered。
        # So far I suppose the cage is a square. So in each direction I use tha max of the corner coordinates as the edeg. Need to update to a smarter version for any shape cage.
        yupper = min(self.box['corners'][:,0])
        ylower = max(self.box['corners'][:,0])
        xleft = min(self.box['corners'][:,1])
        xright = max(self.box['corners'][:,1])
        # print(yupper, ylower, xleft, xright)
        newdata = np.zeros_like(obj_data)
        for i in range(np.shape(obj_data)[0]):
            if obj_data[i,0] <0:
                newdata[i,0] = -obj_data[i,0] / yupper
            elif obj_data[i,0] >= 0:
                newdata[i,0] = obj_data[i,0] / ylower

            if obj_data[i,1] <0:
                newdata[i,1] = -obj_data[i,1] / xleft
            elif obj_data[i,1] >= 0:
                newdata[i,1] = obj_data[i,1] / xright
        
        return(newdata)

    def calculate_object_speed(self, data_relative, savefig = True, figname = 'object_speed_timecourse.svg'):
        # By using the data_relative and cage's size, this function calculate the object 
        # moving speed in the cage.
        r = np.shape(data_relative[0:int(self.fps * self.timecourse_length)])[0]
        newdata = np.zeros(r)
        for i in range(r):
            newdata[i] = math.sqrt((data_relative[i+1, 0] - data_relative[i, 0])**2 + (data_relative[i+1, 1] - data_relative[i, 1])**2) * self.box['edge_length']/2 * self.fps
        
        # Do some post treatment.
        # Smooth the data.
        treateddata = ndimage.gaussian_filter1d(newdata, self.fps) 

        if savefig:
            fig, ax = plt.subplots()
            ax.plot(newdata)
            ax.plot(treateddata)
            ticks, ticklabels = datalength_to_timecourse_label(r,self.fps, 60)
            ax.set_ylabel('speed (m/s)')
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_xlabel('time (min)')
            ax.legend(['raw', 'smoothed'], loc = 1)
            plt.savefig(os.path.join(self.analysis_report_folder, figname))

        return(treateddata)

    def calculate_relative_position_timecourse(self, data_relative, outer_threshold = 0.6, savefig = True, figname = 'object_relative_position_timecourse.svg', **kwargs):
        # Based on the data_relative, this function calculate the object's position in the cage.
        r = np.shape(data_relative[0:int(self.timecourse_length * self.fps)])[0]
        t = np.zeros(r)
        for i in range(r):
            corep = np.argmax(abs(data_relative[i,:]))
            t[i] = abs(data_relative[i,corep])

        if kwargs.get('smooth', False):
            traw = copy.copy(t)
            t = ndimage.gaussian_filter1d(t, self.fps) 

        if savefig:
            fig, ax = plt.subplots()
            if kwargs.get('smooth', False):
                ax.plot(traw * 100)
                ax.plot(t * 100)
                ax.legend(['raw', 'smoothed'], loc = 1)
            else:
                ax.plot(t * 100)

            ax.hlines(outer_threshold * 100, 0,r, linestyles='dotted')
            ticks, ticklabels = datalength_to_timecourse_label(r,self.fps, 60)
            ax.set_ylabel('Position relative to center (%)')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_xlabel('time (min)')
            plt.savefig(os.path.join(self.analysis_report_folder, figname))

        return(t)

    def calculate_cross_times(self, data_relative, outer_threshold = 0.6):
        d = (np.array(data_relative[0:int(self.timecourse_length * self.fps)]) < outer_threshold) * 1
        d = np.concatenate(([0], d, [0])).astype(str)
        d = ''.join(d)
        dreplace = re.sub('01+0', 'Y', d)
        res = len(re.findall('Y', dreplace))
        return(res)

    def plot_position(self, data_relative, filename = 'position_tracking.svg', savefig = True, **kwargs):
        d = data_relative[0:int(self.timecourse_length * self.fps)]
        
        fig, ax = plt.subplots()
        ax.scatter(d[:,1], d[:,0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if savefig:
            plt.savefig(os.path.join(self.analysis_report_folder, filename))

    def analysis_report(self, data_relative, outer_threshold = 0.6, filename = 'result.csv'):
        # This function is a combine of different analyze functions. 
        # It produce several figs and csv data files in the output folder.

        res = {}
        speed = self.calculate_object_speed(data_relative)
        relative_pos = self.calculate_relative_position_timecourse(data_relative, outer_threshold = outer_threshold, smooth = True)
        self.plot_position(data_relative)
        
        res['distance'] = np.sum(speed / self.fps)
        res['outer_time'] = np.sum(relative_pos >= outer_threshold) / self.fps
        res['cross_num'] = self.calculate_cross_times(relative_pos, outer_threshold = outer_threshold)

        df = pd.DataFrame()
        for key, value in res.items():
            df.loc[0,key] = value
        
        df.to_csv(os.path.join(self.analysis_report_folder, filename), index = False)
        return(res)
