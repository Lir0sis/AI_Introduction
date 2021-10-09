import numpy as np
from pandas.core.frame import DataFrame
from tabulate import tabulate

import skfuzzy
import matplotlib.pyplot as plt
import math
import pandas as pd

end = 1#math.pi * 4
points_count = 200

def y(x_value):
    return np.sin(abs(x_value)) * np.cos(x_value/2)

def z(*args):
    if len(args) == 2:
        return np.sin(args[0]) * args[1]
    elif len(args) == 1:
        return y(args[0]) * np.sin(args[0])
    raise KeyError

x = np.linspace(0, end, points_count, dtype='float64')

z_values = [round(z(xn),5) for xn in x]
#z_base = np.linspace(z_values[0], z_values[-1], points_count)

x_max = np.linspace(0, end, 6)
y_max = []
z_max = np.linspace(0, end, 9)

gaussIn = []
gaussOut = []

triIn = []
triOut = []

trapIn = []
trapOut = []

for i in range(6):
    val = x_max[1]
    gaussIn.append(skfuzzy.gaussmf(x, x_max[i], x_max.std()/6))
    triIn.append(skfuzzy.trimf(x, [x_max[i] - val * 3/4, x_max[i], x_max[i] + val * 3/4]))
    trapIn.append(skfuzzy.trapmf(x, [x_max[i] - val * 3/4, x_max[i]- val/4, x_max[i] + val/4, x_max[i] + val * 3/4]))
    plt.plot(x, gaussIn[i])

plt.show()

for i in range(9):
    val = z_max[1]
    gaussOut.append(skfuzzy.gaussmf(x, z_max[i], z_max.std()/9))
    triOut.append(skfuzzy.trimf(x, [z_max[i] - val * 3/4, z_max[i], z_max[i] + val * 3/4]))
    trapOut.append(skfuzzy.trapmf(x, [z_max[i] - val * 3/4, z_max[i]- val/4, z_max[i] + val/4, z_max[i] + val * 3/4]))
    plt.plot(x, gaussOut[i])

plt.show()


def memberFuncValue(input_value, func_values):
    return func_values[round(input_value / (1 / len(func_values)))]

data = {}
data_rules = {}

for xn in x_max:
    y_max.append(round(y(xn), 5))

for n, yn in enumerate(y_max):
    row = []
    for xn in x_max:
        row.append(round(z(xn, yn),5))
    data[yn] = row
    data_rules[f'my{n+1}'] = row.copy()

table1 = pd.DataFrame(data, index=[xn for xn in x_max])
table2 = pd.DataFrame(data_rules, index=[f'mx{n+1}' for n in range(6)])

print(tabulate(table1, headers='keys', tablefmt='psql')) 

for xn in range(len(x_max)):
    for yn in range(len(y_max)):
        cell = table2[f'my{yn+1}'][f'mx{xn+1}']
        f = 0
        for n in range(len(z_max)):
            val1 = memberFuncValue(cell, trapOut[n])
            val2 = memberFuncValue(cell, trapOut[f])
            if max(val1, val2) > val2:
                f = n
        data_rules[f'my{yn+1}'][xn] = f'mf{f}'

table_rules = pd.DataFrame(data_rules, index=[f'mx{n+1}' for n in range(6)])

def getRuleValue(x_values, y_values):
    z_rand = []
    for x_val, y_val in zip(x_values, y_values):
        fx, fy = 0, 0
        for n in range(len(y_max)):
            dy1 =abs(y_max[fy] - y_val)
            dy2 =abs(y_max[n] - y_val)
            if min(dy1, dy2) < dy1:
                fy = n

            dx1 =abs(x_max[fx] - x_val)
            dx2 =abs(x_max[n] - x_val)
            if min(dx1, dx2) < dx1:
                fx = n

        z_rand.append(data[y_max[fy]][fx])
    
    return z_rand


x_rand = np.random.uniform(0, end, points_count)
x_rand.sort()
y_rand = [y(xn) for xn in x_rand]
z_rand = getRuleValue(x_rand, y_rand)
calc_error = 0

for ze, zr in zip(z_values, z_rand):
    calc_error += (abs(ze-zr)/(ze+1)) * 100
calc_error /= len(z_values)

print(tabulate(table_rules, headers='keys', tablefmt='psql'))

print(calc_error, "%")
plt.plot(x, z_values)
plt.plot(x_rand, z_rand)
plt.show()



