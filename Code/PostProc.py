import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

font = {'family': 'serif', 'weight': 'normal', 'size': 18}
FontLegend = {'family': 'serif', 'weight': 'normal', 'size': 15}
plt.rc('font', **font)
colors= [ 'black', 'red', "blue", "royalblue", "lightblue", ]

ColorSpan = ["mistyrose", "lightblue"]

# Read in text file
DirStart = ''
DirList = ["/"]
NameStart = "Sphere"

# Step, Rho, Displacement, Analytical
df1 = pd.read_table('../Results/Sphere/S_140_chi_3.0e-01_G_1e+00_l0_2.0e+00/PostProc.txt', delimiter=' ', header=None)
df1_convert = df1.to_numpy()
# Obtain all components
Step = df1_convert[:,0]
DeltaT = df1_convert[:,1]
Time = df1_convert[:,2]
SurfVal = df1_convert[:,3]
ChemVal = df1_convert[:,4]

# Change according to each file
Step1, Step2, Step3, Step4 = 19, 69, 89, 139
PlotList = [59, 79, 89, 99, 103, 109, 139]

# Figure 1
fig, ax = plt.subplots(1,1, figsize=(5.5,5))
ax.axvspan(0, Step1, edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Step1, Step2, edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)
ax.axvspan(Step2, Step3, edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Step3, Step4, edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)

ax.plot(Step, SurfVal, color="red", linewidth=2, label='Surface Energy')
ax.set_xlabel("Simulation Steps", fontdict=font)
ax.set_ylabel("Surface Energy", color="red",fontdict=font)
# ax.set_ylim((0, SurfMax))

ax2 = ax.twinx()
ax2.plot(Step, ChemVal, color='black', linestyle='--', linewidth=2, label='Chemical Potential')
ax2.set_ylabel("Chemical Potential", color="black", fontdict=font)
ax2.set_xlim((0, Step4))

fig.savefig("../Images/" + NameStart + "SimSteps.pdf", transparent=True, bbox_inches='tight')
plt.close()

# Figure 2
fig, ax = plt.subplots(1,1, figsize=(5.5,5))
ax.axvspan(Time[0], Time[Step1], edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Time[Step1], Time[Step2], edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)
ax.axvspan(Time[Step2], Time[Step3], edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Time[Step3], Time[Step4], edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)
ax.plot(Time, SurfVal, color="red", linewidth=2, label='Surface Energy')
ax.set_xlabel("Time", fontdict=font)
ax.set_ylabel("Surface Energy", color="red",fontdict=font)

ax2 = ax.twinx()
ax2.plot(Time, ChemVal, color='black', linestyle='--', linewidth=2, label='Chemical Potential')
ax2.set_ylabel("Chemical Potential", color="black", fontdict=font)
ax2.set_xscale('log')
fig.savefig("../Images/" + NameStart + "SimTime.pdf", transparent=True, bbox_inches='tight')
plt.close()

DirStart = "../Results/Sphere/"
DirList = ["S_140_chi_3.0e-01_G_1e-02_l0_2.0e+00",
           "S_140_chi_3.0e-01_G_1e-01_l0_2.0e+00",
           "S_140_chi_3.0e-01_G_5e-01_l0_2.0e+00",
           "S_140_chi_3.0e-01_G_1e+00_l0_2.0e+00"]

TitleAxis = ["$\hat{\gamma} = 1 x 10^{-2}$", "$\hat{\gamma} = 1 x 10^{-1}$",
             "$\hat{\gamma} = 5 x 10^{-1}$", "$\hat{\gamma} = 1 x 10^{-0}$"]

fig, ax = plt.subplots(1,1, figsize=(5,5))
DirNo = 0
for Dir in DirList:
    df1 = pd.read_table(DirStart + Dir + '/PostProc.txt', delimiter=' ', header=None)
    df1_convert = df1.to_numpy()
    Time = df1_convert[:,2]
    DispVal = df1_convert[:,5]

    ax.plot(Time, DispVal, linewidth=2, color=colors[DirNo], label=TitleAxis[DirNo])
    ax.set_xscale('log')

    DirNo += 1

ax.legend(loc='upper left', frameon=True, prop=FontLegend)
ax.set_xlabel("Time", fontdict=font)
ax.set_ylabel("Displacement", fontdict=font)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.axvspan(Time[0], Time[Step1], edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Time[Step1], Time[Step2], edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)
ax.axvspan(Time[Step2], Time[Step3], edgecolor=ColorSpan[0], facecolor=ColorSpan[0], alpha=0.5)
ax.axvspan(Time[Step3], Time[Step4], edgecolor=ColorSpan[1], facecolor=ColorSpan[1], alpha=0.5)
ax.set_xlim((Time[0], Time[Step4]))

fig.savefig("../Images/" + NameStart + "DispEvoGv.pdf", transparent=True, bbox_inches='tight')

plt.show()


fig, ax = plt.subplots(1,4, figsize=(18,5))
Points = np.linspace(0, 0.5, 100)         # Points along the profile
DirNo = 0
for Dir in DirList:
    df1 = pd.read_table(DirStart + Dir + '/PostChem.txt', delimiter=' ', header=None)
    df1_convert = df1.to_numpy()

    for i in PlotList:
        ax[DirNo].plot(Points, df1_convert[i,:], linewidth=2, label='{0:.2e}'.format(Time[i]))
    DirNo += 1

for i in [0,1,2,3]:
    ax[i].set_xlim((0,0.5))
    ax[i].set_ylim((-5.5e-3,5e-4))
    ax[i].set_xlabel("Reference Coordinates", fontdict=font)
    ax[i].set_title(TitleAxis[i], fontdict=font)
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax[0].set_ylabel("Chemical Potential", color="black", fontdict=font)
ax[3].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title="Time",frameon=True, prop=FontLegend)

fig.savefig("../Images/" + NameStart + "ChemEvoGv.pdf", transparent=True, bbox_inches='tight')

plt.show()

exit()
ax[1].plot(Time, DispVal, linewidth=2, label='Step {0:.0f}'.format(i))
ax[1].set_xscale('log')
ax[1].set_xlim((10**0, 10**4))


exit()




ax[i].set_title(Title[i] + ": Error {0:.1f}%".format(ErrMeanArr[i]), fontdict=font)
