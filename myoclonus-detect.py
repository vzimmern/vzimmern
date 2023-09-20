import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Defining the columns to read
usecols = ["DIP2_x","DIP2_y","DIP2_z","DIP2_error", "DIP3_x","DIP3_y","DIP3_z","DIP3_error","DIP4_x","DIP4_y","DIP4_z","DIP4_error"]

# Read the CSV file
path_to_csv = "~/Downloads/2019-08-02-vid01.csv"
position_data = pd.read_csv(path_to_csv, usecols=usecols)

# View the first 5 rows
position_data.head()

#time = []
#for i in range(0, len(position_data)):
 #   time.append(i*0.33)   
#position_data['Time'] = time
# position_data.index = np.arange(len(position_data))

dt = 0.033
DIP2_velocity = []


# Using DataFrame.index to calculate velocities
for idx in position_data.index-1:
    DIP2_velocity.append(np.sqrt(np.diff(position_data["DIP2_x"])[idx]**2 + np.diff(position_data["DIP2_y"])[idx]**2 + np.diff(position_data["DIP2_z"])[idx]**2) / dt)
    
velocity_data = pd.DataFrame()
velocity_data["DIP2_vel"] = DIP2_velocity
    

DIP2_accel = []
# Using DataFrame.index to calculate accelerations
for idx in velocity_data.index-1:
    DIP2_accel.append(np.diff(velocity_data["DIP2_vel"])[idx]/dt)

accel_data = pd.DataFrame()
accel_data["DIP2_accel"] = DIP2_accel

DIP2_jerk = []
# Using DataFrame.index to calculate accelerations
for idx in accel_data.index-1:
    DIP2_jerk.append(np.diff(accel_data["DIP2_accel"])[idx]/dt)

jerk_data = pd.DataFrame()
jerk_data["DIP2_jerk"] = DIP2_jerk

jerk_data.plot(y="DIP2_jerk", kind='line')
accel_data.plot(y="DIP2_accel", kind='line')
velocity_data.plot(y="DIP2_vel",kind='line')
