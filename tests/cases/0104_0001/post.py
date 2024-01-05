#!/usr/bin/env python
# coding=utf-8

import pdb
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


res = pd.read_csv("out.csv")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# pdb.set_trace()
inlet = res["name"] == "branch0_seg0"
ax1.plot(res["time"][inlet], res["flow_in"][inlet], "r-")
ax2.plot(res["time"][inlet], res["pressure_in"][inlet], "b-")
# ax2.plot(res["time"][inlet], res["d_pressure_in"][inlet], "g-")

outlet = res["name"] == "branch4_seg2"

p = res["pressure_in"][inlet].to_list()
dp = res["d_pressure_in"][inlet].to_list()
q = res["flow_out"][outlet].to_list()
print(p)
print([np.max(p), np.min(p), np.max(dp)])

plt.show()
