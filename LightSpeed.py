import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from scipy import interpolate
import scipy
import os
from scipy.optimize import curve_fit
from scipy.integrate import simps
import pandas as pd

time_differences_all = []
#Nested loop so that you can do process for each distance
#and each frequency
#So you have a list of lists, where each list inside the bigger list
#is the time shift for each frequency at a distance
#later we take the average of each of these
#so you get the average time shift for a distance
#which gives you the five values we plot at the end
for x in [18,24,30,36]:
    x = str(x)
    time_difs = []
    for y in [4,5,6,7,8]:
        y = str(y)
        curve1 = pd.read_csv("C:/Users/suyas/github/Light Speed/New_Light/" + x + "/" + y + "/ALL0000/F0000CH1.csv",header=None)
        curve2 = pd.read_csv("C:/Users/suyas/github/Light Speed/New_Light/" + x + "/" + y + "/ALL0000/F0000CH2.csv",header=None)

        #making them on the same scale from the data
        curve2_scale = float(curve2[1][8])/float(curve1[1][8])

        #I just multiplied by 10**7 to get better intuition about the plots
        #I undo it later by multiplying by 10**-7
        plt.plot(curve1[3], curve1[4])
        plt.plot(curve2[3], curve2[4]/curve2_scale)

        plt.ylim(-6,6)
        plt.xlim(-3,3)

        g = np.where(curve1[4] == 0)
        f = np.where(curve2[4] == 0)

        plt.plot(curve1[3][g[0]],curve1[3][g[0]],'bo')
        plt.plot(curve2[3][f[0]],curve2[3][f[0]],'ro')

        plt.xlabel('$/Delta$t')
        plt.title('Phase Shift of the Waves ' + x + ' feet and ' + y + ' MHz')
        plt.savefig('Phase Shift of the Waves ' + x + ' feet and ' + y + ' MHz')
        #plt.show()
        for n in [0]:
            shift = np.abs(curve2[3][f[0][n]] - curve1[3][g[0][n]])
            time_difs = np.append(time_difs,shift)   #
    time_differences_all.append(time_difs)

print(time_differences_all)


#Doing what i talked about at the beginning
time_differences_averages = []
stda = []
for x in time_differences_all:
    mean = np.average(x, weights=[0.1,1,0.2,1,0.2])
    std = np.std(x)
    time_differences_averages.append(mean)
    stda.append(std)

print("average",time_differences_averages)
print("standard",stda)
k=[18*0.3,24*0.3,30*0.3,36*0.3]
re =[40*0.3,36*0.3,30*0.3,24*0.3,18*0.3]
dist_m = np.array(k)
fig0 = plt.figure(figsize=(8,4))
fig0.patch.set_facecolor('xkcd:black')
plt.style.use('dark_background')
ax0=plt.subplot(1,2,1)

#plt.xlabel('$\Delta$ t')
#plt.ylabel('Distance in meters')

#plt.show()




theory = (dist_m)/2.99702547
exact = np.array(time_differences_averages)#*10**8
print("theory",theory)
print("exact",exact)



d = ((exact-theory)**2)/theory


print(sum(d))


ax0.plot(dist_m,exact,'bo')






from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from pprint import pprint

from iminuit import Minuit, __version__
print('iminuit version', __version__)

data_x = dist_m
data_y = exact
ysig = np.array(stda)#*10**8
#errorbar(data_x,data_y,yerr=ysig,fmt="o",color='k',solid_capstyle='projecting',capsize=5)

#print(data_y)
#def model(x, a, b, c,d):
def model(x, a):
    return x*(1/a)

def least_squares(a):
#def least_squares(a,b,c,d):
    return sum(((data_y - model(data_x,a))**2 / ysig**2 ))#
    #return sum((data_y - model(data_x,a,b,c,d))**2 / ysig**2)



m = Minuit(least_squares, 2.99702547 )
#m = Minuit(least_squares, 200,180,0.45,0.55)
#m = Minuit(least_squares, 5.29,5.01,2.0,2.5)
m.migrad() # finds minimum of least_squares function
m.hesse() # computes errors
print(' ')
print(' Covariance matrix')
#pprint(m.covariance())
print(' ')
#plt.plot(model(data_x, *m.values), 'b-')
plt.plot(data_x, model(data_x, *m.values), 'r-')
print(' ')
print(' Fitting parameters')
for p in m.parameters:
    print("{} = {} +- {}".format(p,m.values[p], m.errors[p]))

print(' ')
print('chi square:', m.fval)
print('number of degrees of freedom:', (len(data_y)- len(m.parameters)))
print('reduced chi square:', m.fval / (len(data_y) - len(m.parameters)) )
print(len(m.parameters))
print(m.parameters)
show()
