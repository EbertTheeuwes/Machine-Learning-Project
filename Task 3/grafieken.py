import random
import time
import matplotlib.pyplot as plt
import math
import pyspiel
import numpy as np


y_template = np.array([0.000308990478515625,0.033883094787597656,23.028602600097656])
y_advanced = np.array([0.0003154277801513672, 0.002176523208618164, 0.022825241088867188, 0.060465335845947266, 0.22223877906799316, 5.397651433944702])
x_template = np.array([0,1,2])
x_advanced = np.array([0,1,2,3,4,5])
my_xticks_template = ['1X1','2X1','3X1']
my_xticks_advanced = ['1X1','2X1','3X1','2X2', '4X1', '3X2']
sizes = np.array([640, 4696, 36960, 147552, 295000, 10485856])
keys = np.array([15, 139, 1208, 5109, 10228, 180201])
plt.xticks(x_template, my_xticks_template)
plt.xticks(x_advanced, my_xticks_advanced)
#plt.plot(x_template, y_template)
plt.plot(x_advanced, y_advanced)
plt.ylabel('time [s]')
plt.xlabel('grid size')
plt.legend()
plt.show()







#times = []
#x_values = []
#special_values = [0]
#for n in range(150):
#    cost = 0
#    x_values += [n]
#    if n != 0:
#        special_values += [0.0000008 * n * math.log(n, 10)]
#        #special_values += [0.00000012 * n**2]
#    for i in range(800):
        #randomL = randomList(n)
#        randomL = []
#        for k in range(n):
#            randomL += [n-k]
        #print('unordered', randomL)
#        start = time.time()
#        quickSort(randomL, 0, n - 1)
#        end = time.time()
#        cost += end - start
        #print('ordered', randomL)
#    avg_time = cost / 800
#    times += [avg_time]
#print(times)
#plt.plot(x_values, times, label='cost')
#plt.plot([x_values[0], x_values[-1]], [times[0], times[-1]], label='O(n)')
#plt.plot(x_values, special_values, label='O(n*ln(n))')
#plt.plot(x_values, special_values, label='O(n^2)')
#plt.ylabel('time[s]')
#plt.xlabel('n')
#plt.legend()
#plt.show()