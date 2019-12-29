import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# x = [1,2,3,4,5,6,7,8,9,10]
# y = [18.72, 18.58, 18.46, 18.36, 18.26, 18.22, 18.16, 18.11, 18.07, 18.04]
# plt.plot(x,y)
# plt.show()
# y1 = [18.43, 18.17, 18.03,17.95,17.91,17.88,17.87,17.86,17.86,17.87]
# plt.plot(x,y1)
# plt.show()
# y2 = [18.8,18.72,18.65,18.59,18.53,18.47,18.42,18.37,18.33,18.30]
# plt.plot(x,y2)
# plt.show()
# x3 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# y3 = [18.57,18.35,18.20,18.09,18.02,17.97,17.93,17.91,17.89,17.88,17.875,17.872,17.871,17.871]
# plt.plot(x3,y3)
# plt.show()
trainmse = [18.19,17.89,17.9,18.05,18.53,18.88,17.27]
traintime = [2.94,3.1,3,2.9,3.4,51.61,0.004]
plt.scatter(traintime,trainmse)
plt.show()
testmse = [19.35,19.48,19.47,19.38,19.32,19.35,20.02]
testtime = [2.94,3.1,3,2.9,3.4,51.61,0.004]
plt.scatter(testtime,testmse)
plt.show()

