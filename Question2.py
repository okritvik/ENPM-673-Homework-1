"""
@author Kumara Ritvik Oruganti

"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def get_top_bottom_pixels(red,shape,indices):
    height = []
    width = []
    #'''
    indices_array = np.where(red<230)
    height = list(indices_array[0])
    width = list(indices_array[1])
    #print(height)
    #print(width)
    '''
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            if(red[i][j]<230):
                height.append(i)
                width.append(j)
                #print("Appended")'''
    indices.append((((shape[0]-min(height))+(shape[0]-max(height)))/2,((width[height.index(min(height))])+(width[height.index(max(height))]))/2))
    
def least_squares_curve_fitting(width_indices,height_indices):
    sx4, sx3, sx2, sx, s, syx2, syx, sy = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(0,len(width_indices)):
        sx4 = sx4 + width_indices[i]**4
        sx3 = sx3 + width_indices[i]**3
        sx2 = sx2 + width_indices[i]**2
        sx = sx + width_indices[i]
        s = s + 1
        syx2 = syx2 + height_indices[i]*(width_indices[i]**2)
        syx = syx + height_indices[i]*width_indices[i]
        sy = sy + height_indices[i]

    A = np.array([(sx4, sx3, sx2),
                  (sx3,sx2,sx),
                  (sx2,sx,s)])
    B = np.array([(syx2,syx,sy)]).transpose()
    A_inv = np.linalg.inv(A)
    X = np.matmul(A_inv,B)
    # print(X)
    return X
def plot_raw(width_indices,height_indices):
    fig,ax = plt.subplots()
    ax.scatter(width_indices,height_indices)
    ax.set_title("Original Trajectory")
    
def plot_curve(X,width_indices,height_indices,ax):
    x = np.linspace(0,max(width_indices)+50,10000)
    y = X[0]*(x**2) + X[1]*x + X[2]
    ax.plot(x,y,'r')
    ax.scatter(width_indices,height_indices)

def get_height_width(indices,height_indices,width_indices):
    for i in range(0,len(indices)):
        height_indices.append(indices[i][0])
        width_indices.append(indices[i][1])
    
def video_1():
    path = os.getcwd()
    print(path)

    video = cv2.VideoCapture(path+"/ball_video1.mp4")
    #Checking if the video is opened

    if(video.isOpened()==False):
        print("Error opening the video. Check the path")
    else:
        print("Video opened")

    indices = []
    #Reading the frames
    while(video.isOpened()):
        ret,frame = video.read()
        if ret:
            shape = frame.shape
            red = frame[0:shape[0],0:shape[1],2]
            get_top_bottom_pixels(red,shape,indices)
        else:
            print("Video Ended")
            break
    video.release()
    print("Destroyed All Windows")
    height_indices = []
    width_indices = []
    get_height_width(indices,height_indices,width_indices)
    plot_raw(width_indices, height_indices)
    #Now calculating the required parameters for curve fitting
    X = least_squares_curve_fitting(width_indices,height_indices)
    fig,ax = plt.subplots()
    plot_curve(X,width_indices,height_indices,ax)
    ax.set_title("Curve Fitting for the Video 1 Trajectory")
    
def video_2():
    path = os.getcwd()
    print(path)

    video = cv2.VideoCapture(path+"/ball_video2.mp4")
    #Checking if the video is opened

    if(video.isOpened()==False):
        print("Error opening the video. Check the path")
    else:
        print("Video opened")

    indices = []
    #Reading the frames
    while(video.isOpened()):
        ret,frame = video.read()
        if ret:
            shape = frame.shape
            red = frame[0:shape[0],0:shape[1],2]
            get_top_bottom_pixels(red,shape,indices)
        else:
            print("Video Ended")
            break
    video.release()
    print("Destroyed All Windows")
    height_indices = []
    width_indices = []
    get_height_width(indices,height_indices,width_indices)
    plot_raw(width_indices, height_indices)
    #Now calculating the required parameters for curve fitting
    X = least_squares_curve_fitting(width_indices,height_indices)
    fig,ax = plt.subplots()
    plot_curve(X,width_indices,height_indices,ax)
    ax.set_title("Curve Fitting for the Video 2 Trajectory")

def main():
    video_1()
    video_2()
    plt.show()

if __name__ == '__main__':
    main()