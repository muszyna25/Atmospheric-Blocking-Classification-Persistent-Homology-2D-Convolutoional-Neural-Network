#!/usr/bin/env python

import numpy as np

TX = range(106,160)

DEG = [np.around(((360.0/tx)*0.33), decimals=4) for tx in TX]

for i in range(0,len(DEG)): print('%i: %i, %0.4f' %(i, TX[i], DEG[i]))

