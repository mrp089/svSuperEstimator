#!/usr/bin/env python
# coding=utf-8

import pandas as pd

res = pd.read_csv("spitieris22_inflow.csv")

print(res.t.tolist())
print(res.Q.tolist())
