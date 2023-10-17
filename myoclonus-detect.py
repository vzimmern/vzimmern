import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
path_to_csv = "~/Documents/CSTB-mouse-trial-Vincent Zimmern-2023-08-16/videos/CSTB-mouse-below-2DLC_resnet50_CSTB-mouse-trialAug16shuffle1_500000.csv"
position_data = pd.read_csv(path_to_csv)

dt = 0.033
H_vel = []
RA_vel = []
LA_vel = []
RL_vel = []
LL_vel = []
B_vel = []
TB_vel = []
TE_vel = []

position_data.columns = ["time","head-x", "head-y", "head-prob","RA-x","RA-y","RA-prob","LA-x","LA-y","LA-prob","RL-x","RL-y","RL-prob","LL-x","LL-y","LL-prob","body-x","body-y","body-prob","TB-x","TB-y","TB-prob", "TE-x","TE-y", "TE-prob"]
position_data = position_data.drop([0,1], axis=0)
position_data = position_data.reset_index(drop=True)

for idx in position_data.index-1:
    H_vel.append(np.sqrt(np.diff(position_data["H_x"])[idx]**2 + np.diff(position_data["H_y"])[idx]**2)/dt)
    RA_vel.append(np.sqrt(np.diff(position_data["RA_x"])[idx]**2 + np.diff(position_data["RA_y"])[idx]**2)/dt)
    LA_vel.append(np.sqrt(np.diff(position_data["LA_x"])[idx]**2 + np.diff(position_data["LA_y"])[idx]**2)/dt)
    RL_vel.append(np.sqrt(np.diff(position_data["RL_x"])[idx]**2 + np.diff(position_data["RL_y"])[idx]**2)/dt)
    LL_vel.append(np.sqrt(np.diff(position_data["LL_x"])[idx]**2 + np.diff(position_data["LL_y"])[idx]**2)/dt)
    B_vel.append(np.sqrt(np.diff(position_data["B_x"])[idx]**2 + np.diff(position_data["B_y"])[idx]**2)/dt)
    TB_vel.append(np.sqrt(np.diff(position_data["TB_x"])[idx]**2 + np.diff(position_data["TB_y"])[idx]**2)/dt)
    TE_vel.append(np.sqrt(np.diff(position_data["TE_x"])[idx]**2 + np.diff(position_data["TE_y"])[idx]**2)/dt)

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

H_accel = []
RA_accel = []
RL_accel = []
LA_accel = []
LL_accel = []
B_accel = []
TB_accel = []
TE_accel = []

# Using DataFrame.index to calculate accelerations
for idx in df.index+1:
    H_accel.append(np.diff(df["H_vel"])[idx]/dt)
    RA_accel.append(np.diff(df["RA_vel"])[idx]/dt)
    RL_accel.append(np.diff(df["RL_vel"])[idx]/dt)
    LA_accel.append(np.diff(df["LA_vel"])[idx]/dt)
    LL_accel.append(np.diff(df["LL_vel"])[idx]/dt)
    B_accel.append(np.diff(df["B_vel"])[idx]/dt)
    TB_accel.append(np.diff(df["TB_vel"])[idx]/dt)
    TE_accel.append(np.diff(df["TE_vel"])[idx]/dt)
    
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

H_jerk = []
RA_jerk = []
RL_jerk = []
LA_jerk = []
LL_jerk = []
B_jerk = []
TB_jerk = []
TE_jerk = []

# Using DataFrame.index to calculate accelerations
for idx in df.index-1:
    H_jerk.append(abs(np.diff(df["H_accel"])[idx]/dt))
    RA_jerk.append(abs(np.diff(df["RA_accel"])[idx]/dt))
    RL_jerk.append(abs(np.diff(df["RL_accel"])[idx]/dt))
    LA_jerk.append(abs(np.diff(df["LA_accel"])[idx]/dt))
    LL_jerk.append(abs(np.diff(df["LL_accel"])[idx]/dt))
    B_jerk.append(abs(np.diff(df["B_accel"])[idx]/dt))
    TB_jerk.append(abs(np.diff(df["TB_accel"])[idx]/dt))
    TE_jerk.append(abs(np.diff(df["TE_accel"])[idx]/dt))
    
df.insert(6, "H_jerk", H_jerk, True)
df.insert(12, "RA_jerk", RA_jerk, True)
df.insert(18, "RL_jerk", RL_jerk, True)
df.insert(24, "LA_jerk", LA_jerk, True)
df.insert(30, "LL_jerk", LL_jerk, True)
df.insert(36, "B_jerk", B_jerk, True)
df.insert(42, "TB_jerk", TB_jerk, True)
df.insert(48, "TE_jerk", TE_jerk, True)

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

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["H_vel"])
axs[0, 0].set_title("Head velocity")
axs[1, 0].plot(df["time"], df["H_accel"], 'tab:orange')
axs[1, 0].set_title("Head acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["H_jerk"], 'tab:green')
axs[0, 1].set_title("Head jerk")
axs[1, 1].plot(df["time"], df["H_prob"], 'tab:red')
axs[1, 1].set_title("Head position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["RA_vel"])
axs[0, 0].set_title("Right arm velocity")
axs[1, 0].plot(df["time"], df["RA_accel"], 'tab:orange')
axs[1, 0].set_title("Right arm acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["RA_jerk"], 'tab:green')
axs[0, 1].set_title("Right arm jerk")
axs[1, 1].plot(df["time"], df["RA_prob"], 'tab:red')
axs[1, 1].set_title("Right arm position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["LA_vel"])
axs[0, 0].set_title("Left arm velocity")
axs[1, 0].plot(df["time"], df["LA_accel"], 'tab:orange')
axs[1, 0].set_title("Left arm acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["LA_jerk"], 'tab:green')
axs[0, 1].set_title("Left arm jerk")
axs[1, 1].plot(df["time"], df["LA_prob"], 'tab:red')
axs[1, 1].set_title("Left arm position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["RL_vel"])
axs[0, 0].set_title("Right leg velocity")
axs[1, 0].plot(df["time"], df["RL_accel"], 'tab:orange')
axs[1, 0].set_title("Right leg acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["RL_jerk"], 'tab:green')
axs[0, 1].set_title("Right leg jerk")
axs[1, 1].plot(df["time"], df["RL_prob"], 'tab:red')
axs[1, 1].set_title("Right leg position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["LL_vel"])
axs[0, 0].set_title("Left leg velocity")
axs[1, 0].plot(df["time"], df["LL_accel"], 'tab:orange')
axs[1, 0].set_title("Left leg acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["LL_jerk"], 'tab:green')
axs[0, 1].set_title("Left leg jerk")
axs[1, 1].plot(df["time"], df["LL_prob"], 'tab:red')
axs[1, 1].set_title("Left leg position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["B_vel"])
axs[0, 0].set_title("Body velocity")
axs[1, 0].plot(df["time"], df["B_accel"], 'tab:orange')
axs[1, 0].set_title("Body acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["B_jerk"], 'tab:green')
axs[0, 1].set_title("Body jerk")
axs[1, 1].plot(df["time"], df["B_prob"], 'tab:red')
axs[1, 1].set_title("Body position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["TB_vel"])
axs[0, 0].set_title("Tail base velocity")
axs[1, 0].plot(df["time"], df["TB_accel"], 'tab:orange')
axs[1, 0].set_title("Tail base acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["TB_jerk"], 'tab:green')
axs[0, 1].set_title("Tail base jerk")
axs[1, 1].plot(df["time"], df["TB_prob"], 'tab:red')
axs[1, 1].set_title("Tail base position probability")
fig.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df["time"], df["TE_vel"])
axs[0, 0].set_title("Tail end velocity")
axs[1, 0].plot(df["time"], df["TE_accel"], 'tab:orange')
axs[1, 0].set_title("Tail end acceleration")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(df["time"], df["TE_jerk"], 'tab:green')
axs[0, 1].set_title("Tail end jerk")
axs[1, 1].plot(df["time"], df["TE_prob"], 'tab:red')
axs[1, 1].set_title("Tail end position probability")
fig.tight_layout()

average_uncertainty = (df["H_prob"] + df["RA_prob"] + df["RL_prob"] +  df["RL_prob"] + df["LL_prob"] + df["B_prob"]  + df["TB_prob"] + df["TE_prob"])/8
average_jerk = (df["H_jerk"] + df["RA_jerk"] + df["RL_jerk"] +  df["RL_jerk"] + df["LL_jerk"] + df["B_jerk"]  + df["TB_jerk"] + df["TE_jerk"])/8
df.insert(49, "Average Jerk", average_jerk, True)
df.insert(50, "Average Uncertainty", average_uncertainty, True)

fig, axs = plt.subplots(2)
axs[0].plot(df["time"], df["Average Jerk"], 'tab:orange')
axs[0].set_title("Average jerk")
axs[1].plot(df["time"], df["Average Uncertainty"], 'tab:green')
axs[1].set_title("Average position uncertainty")
