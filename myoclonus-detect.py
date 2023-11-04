import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt 
import matplotlib.animation as animation 
from matplotlib.animation import FuncAnimation
from IPython import display

import moviepy.editor as mp
from moviepy.editor import * 

# Read the CSV file: set the path to where the CSV file is located
path_to_csv = "/home/minassian/Documents/EPM1-10-20-23-Vincent Zimmern-2023-10-20/videos/EPM1-09-2023-below-combinedDLC_resnet50_EPM1-10-20-23Oct20shuffle1_500000.csv"

# Path to find the labelled video corresponding to the CSV file
path_to_vid = "/home/minassian/Documents/EPM1-10-20-23-Vincent Zimmern-2023-10-20/videos/EPM1-09-2023-below-combinedDLC_resnet50_EPM1-10-20-23Oct20shuffle1_500000_labeled.mp4"

# paths where you would like to store the average jerk, uncertainty, and head jerk of your recording -- please adjust the MP4 name to reflect the specific experiment/recording (e.g. WT-S100-11-3-2023 or EPM1-S121-11-2-2023)
path_to_jerk =  "/home/minassian/Documents/Jerk and Uncertainty Calculations/jerk_animation.mp4"
path_to_uncert = "/home/minassian/Documents/Jerk and Uncertainty Calculations/uncertainty_animation.mp4"
path_to_h_jerk = "/home/minassian/Documents/Jerk and Uncertainty Calculations/head_jerk_animation.mp4"

df = pd.read_csv(path_to_csv)

# define time parameter from the frame rate (30 FPS for most uses, might be 60 FPS)
dt = 0.033

# define empty lists for the body parts
# i.e. H = head, RA = right arm, RL = right leg, B = body, TB = tail base, TE = tail end
time = []
H_vel = []
RA_vel = []
LA_vel = []
RL_vel = []
LL_vel = []
B_vel = []
TB_vel = []
TE_vel = []

# reassign column names for convenience in the dataframe
df.columns = ["time","H_x", "H_y", "H_prob","RA_x","RA_y","RA_prob","LA_x","LA_y","LA_prob","RL_x","RL_y","RL_prob","LL_x","LL_y","LL_prob","B_x","B_y","B_prob","TB_x","TB_y","TB_prob","TE_x","TE_y","TE_prob"]

# clean up the dataframe and convert to floats for computation
df = df.drop([0,1], axis=0)
df = df.reset_index(drop=True)
df = df.astype(float)

# convert frame number to approximate time in seconds
for idx in df.index:
    time.append(idx * dt)
    
# calculate velocities from pixel positions (has to be up to index-1 because diff requires two values)   
for idx in df.index-1:
    H_vel.append(np.sqrt(np.diff(df["H_x"])[idx]**2 + np.diff(df["H_y"])[idx]**2)/dt)
    RA_vel.append(np.sqrt(np.diff(df["RA_x"])[idx]**2 + np.diff(df["RA_y"])[idx]**2)/dt)
    LA_vel.append(np.sqrt(np.diff(df["LA_x"])[idx]**2 + np.diff(df["LA_y"])[idx]**2)/dt)
    RL_vel.append(np.sqrt(np.diff(df["RL_x"])[idx]**2 + np.diff(df["RL_y"])[idx]**2)/dt)
    LL_vel.append(np.sqrt(np.diff(df["LL_x"])[idx]**2 + np.diff(df["LL_y"])[idx]**2)/dt)
    B_vel.append(np.sqrt(np.diff(df["B_x"])[idx]**2 + np.diff(df["B_y"])[idx]**2)/dt)
    TB_vel.append(np.sqrt(np.diff(df["TB_x"])[idx]**2 + np.diff(df["TB_y"])[idx]**2)/dt)
    TE_vel.append(np.sqrt(np.diff(df["TE_x"])[idx]**2 + np.diff(df["TE_y"])[idx]**2)/dt)

# assign time and velocities as new columns in the dataframe
df["time"] = time
df.insert(4, "H_vel", H_vel, True)
df.insert(8, "RA_vel", RA_vel, True)
df.insert(12, "RL_vel", RL_vel, True)
df.insert(16, "LA_vel", LA_vel, True)
df.insert(20, "LL_vel", LL_vel, True)
df.insert(24, "B_vel", B_vel, True)
df.insert(28, "TB_vel", TB_vel, True)
df.insert(32, "TE_vel", TE_vel, True)

# set the 1st value of the velocity columns to 0 because the 1st entry in that column doesn't correspond to a physical value
df["H_vel"][0] = 0
df["RA_vel"][0] = 0
df["RL_vel"][0] = 0
df["LA_vel"][0] = 0
df["LL_vel"][0] = 0
df["B_vel"][0] = 0
df["TB_vel"][0] = 0
df["TE_vel"][0] = 0

# create new lists for body part acceleration
H_accel = []
RA_accel = []
RL_accel = []
LA_accel = []
LL_accel = []
B_accel = []
TB_accel = []
TE_accel = []

# Using DataFrame.index to calculate accelerations (same as velocity)
for idx in df.index-1:
    H_accel.append(np.diff(df["H_vel"])[idx]/dt)
    RA_accel.append(np.diff(df["RA_vel"])[idx]/dt)
    RL_accel.append(np.diff(df["RL_vel"])[idx]/dt)
    LA_accel.append(np.diff(df["LA_vel"])[idx]/dt)
    LL_accel.append(np.diff(df["LL_vel"])[idx]/dt)
    B_accel.append(np.diff(df["B_vel"])[idx]/dt)
    TB_accel.append(np.diff(df["TB_vel"])[idx]/dt)
    TE_accel.append(np.diff(df["TE_vel"])[idx]/dt)
    
# assign accelerations to the relevant body parts as new columns
df.insert(5, "H_accel", H_accel, True)
df.insert(10, "RA_accel", RA_accel, True)
df.insert(15, "RL_accel", RL_accel, True)
df.insert(20, "LA_accel", LA_accel, True)
df.insert(25, "LL_accel", LL_accel, True)
df.insert(30, "B_accel", B_accel, True)
df.insert(35, "TB_accel", TB_accel, True)
df.insert(40, "TE_accel", TE_accel, True)

# setting first two values of acceleration at 0 because not corresponding to physical values
df["H_accel"][0] = 0
df["H_accel"][1] = 0
df["RA_accel"][0] = 0
df["RA_accel"][1] = 0
df["RL_accel"][0] = 0
df["RL_accel"][1] = 0
df["LA_accel"][0] = 0
df["LA_accel"][1] = 0
df["LL_accel"][0] = 0
df["LL_accel"][1] = 0
df["B_accel"][0] = 0
df["B_accel"][1] = 0
df["TB_accel"][0] = 0
df["TB_accel"][1] = 0
df["TE_accel"][0] = 0
df["TE_accel"][1] = 0

# create new lists for jerk 
H_jerk = []
RA_jerk = []
RL_jerk = []
LA_jerk = []
LL_jerk = []
B_jerk = []
TB_jerk = []
TE_jerk = []

# Using DataFrame.index to calculate jerks
for idx in df.index-1:
    H_jerk.append(abs(np.diff(df["H_accel"])[idx]/dt))
    RA_jerk.append(abs(np.diff(df["RA_accel"])[idx]/dt))
    RL_jerk.append(abs(np.diff(df["RL_accel"])[idx]/dt))
    LA_jerk.append(abs(np.diff(df["LA_accel"])[idx]/dt))
    LL_jerk.append(abs(np.diff(df["LL_accel"])[idx]/dt))
    B_jerk.append(abs(np.diff(df["B_accel"])[idx]/dt))
    TB_jerk.append(abs(np.diff(df["TB_accel"])[idx]/dt))
    TE_jerk.append(abs(np.diff(df["TE_accel"])[idx]/dt))
    
# assign jerk values to the dataframe as new columns    
df.insert(6, "H_jerk", H_jerk, True)
df.insert(12, "RA_jerk", RA_jerk, True)
df.insert(18, "RL_jerk", RL_jerk, True)
df.insert(24, "LA_jerk", LA_jerk, True)
df.insert(30, "LL_jerk", LL_jerk, True)
df.insert(36, "B_jerk", B_jerk, True)
df.insert(42, "TB_jerk", TB_jerk, True)
df.insert(48, "TE_jerk", TE_jerk, True)

# set the first 3 values of jerk column to 0 as jerk can only be computed with 1st 3 position (pixel) values
df["H_jerk"][0] = 0
df["H_jerk"][1] = 0
df["H_jerk"][2] = 0

df["RA_jerk"][0] = 0
df["RA_jerk"][1] = 0
df["RA_jerk"][2] = 0

df["RL_jerk"][0] = 0
df["RL_jerk"][1] = 0
df["RL_jerk"][2] = 0

df["LA_jerk"][0] = 0
df["LA_jerk"][1] = 0
df["LA_jerk"][2] = 0

df["LL_jerk"][0] = 0
df["LL_jerk"][1] = 0
df["LL_jerk"][2] = 0

df["B_jerk"][0] = 0
df["B_jerk"][1] = 0
df["B_jerk"][2] = 0

df["TB_jerk"][0] = 0
df["TB_jerk"][1] = 0
df["TB_jerk"][2] = 0

df["TE_jerk"][0] = 0
df["TE_jerk"][1] = 0
df["TE_jerk"][2] = 0

# create average jerk and average uncertainty 
average_uncertainty = (df["H_prob"] + df["RA_prob"] + df["RL_prob"] +  df["RL_prob"] + df["LL_prob"] + df["B_prob"]  + df["TB_prob"] + df["TE_prob"])/8
average_jerk = (df["H_jerk"] + df["RA_jerk"] + df["RL_jerk"] +  df["RL_jerk"] + df["LL_jerk"] + df["B_jerk"]  + df["TB_jerk"] + df["TE_jerk"])/8

# averaging dataframe columns leads to Series objects. This step converts Series to lists
average_uncertainty = average_uncertainty.tolist()
average_jerk = average_jerk.tolist()

# assign average jerk and uncerainty to dedicated columns in dataframe
df.insert(49, "Average Jerk", average_jerk, True)
df.insert(50, "Average Uncertainty", average_uncertainty, True)

# Create a VideoCapture object from myoclonus videos
cap = cv2.VideoCapture(path_to_vid)

# check that video is opened correctly
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# make sure frame rate is correct
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate: ", int(fps), "FPS")

# -------------------------- Generate animations for average jerk, head jerk, and uncertainty -----

# --------------------- Generate animation of average jerk over time ------------------------

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-')
fig.suptitle("Average jerk")

import time

tic = time.perf_counter()


def init():
    ax.grid()
    ax.set_xlim(0, df["time"].max())
    ax.set_ylim(0, df["Average Jerk"].max())
    return ln, 

def update(frame):
    ax.set(xlabel="Video time (sec)", ylabel="Jerk (pixels/second cubed)")
    xdata.append(df.iat[frame, 0])
    ydata.append(df.iat[frame, 49])
    ln.set_data(xdata, ydata)
    ax.set_title(f"N={frame}")
    ax.autoscale_view(True, True)
    ax.relim()
    return ln,

ani = FuncAnimation(fig, update, frames=len(df), init_func= init, interval=30, blit=True)


# please fill out the path below to which you would want to save the jerk animation
ani.save(path_to_jerk, fps=30, extra_args=['-vcodec', 'libx264'])
toc = time.perf_counter()

print(f"Video was generated in {(toc - tic)/60:0.4f} minutes")
 
# ------------------------ animation for averaged uncertainty of limb, head, body position ---------------

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'g-')
fig.suptitle("Average uncertainty")

tic = time.perf_counter()

def init():
    ax.grid()
    ax.set_xlim(0, df["time"].max())
    ax.set_ylim(0, df["Average Uncertainty"].max())
    return ln, 

def update(frame):
    ax.set(xlabel="Video time (sec)", ylabel="Uncertainty (1=100%)")
    xdata.append(df.iat[frame, 0])
    ydata.append(df.iat[frame, 50])
    ln.set_data(xdata, ydata)
    ax.set_title(f"N={frame}")
    ax.autoscale_view(True, True)
    ax.relim()
    return ln,

ani = FuncAnimation(fig, update, frames=len(df), init_func= init, interval=30, blit=True)

# please fill out the path below to which you would want to save the uncertainty animation
ani.save(path_to_uncert, fps=30, extra_args=['-vcodec', 'libx264'])
toc = time.perf_counter()

print(f"Video was generated in {(toc - tic)/60:0.4f} minutes")

 
# ------------------------ animation for head jerk over time ---------------

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'b-')
fig.suptitle("Head jerk")

tic = time.perf_counter()

def init():
    ax.grid()
    ax.set_xlim(0, df["time"].max())
    ax.set_ylim(0, df["H_jerk"].max())
    return ln, 

def update(frame):
    ax.set(xlabel="Video time (sec)", ylabel="Head jerk (pixels/second cubed)")
    xdata.append(df.iat[frame, 0])
    ydata.append(df.iat[frame, 6])
    ln.set_data(xdata, ydata)
    ax.set_title(f"N={frame}")
    ax.autoscale_view(True, True)
    ax.relim()
    return ln,

ani = FuncAnimation(fig, update, frames=len(df), init_func= init, interval=30, blit=True)
ani.save(path_to_h_jerk, fps=30, extra_args=['-vcodec', 'libx264'])
toc = time.perf_counter()

print(f"Video was generated in {(toc - tic)/60:0.4f} minutes")

# ------------ Generating combined video of average jerk, head jerk, uncertainty, and footage ---

# make sure that your underlying Python shell has moviepy module installed
# you may need to go into terminal and type "pip install moviepy"

# loading mouse footage
mouse_footage = mp.VideoFileClip(path_to_vid).margin(10) 

# loading jerk animation
jerk_animation = mp.VideoFileClip(path_to_jerk).margin(10) 
 
# loading uncertainty/likelihood animation
uncert_animation = mp.VideoFileClip(path_to_uncert).margin(10) 
 
# loading head jerk animation
head_jerk_animation = mp.VideoFileClip(path_to_h_jerk).margin(10)

# clip list
clips = [mouse_footage, jerk_animation, uncert_animation, head_jerk_animation]

mouse_footage.set_position(("left","center"))
jerk_animation.set_position(("right", "top"))
uncert_animation.set_position(("right", "middle"))
head_jerk_animation.set_position("right", "bottom")

# from moviepy.editor import VideoFileClip, clips_array, vfx
final_clip = clips_array([[mouse_footage, jerk_animation],
                          [uncert_animation, head_jerk_animation]])

final_clip.resize(width=480).write_videofile("stacked_jerk_uncert_footage_resized.mp4")



