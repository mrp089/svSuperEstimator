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
ax1.plot(res["time"], res["flow_in"], "r-")
ax2.plot(res["time"], res["pressure_in"], "b-")

p = res["pressure_in"].to_list()
print(p)
print([np.max(p), np.min(p)])

plt.show()
