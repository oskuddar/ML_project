
# ML Packages
# Interpreter: conda activate ml_env

def tokenize (text):
    return text.split(" ")

text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley"
print(tokenize(text))

# Numpy

import numpy as np

a_py = [1,2,3,4,5,6,7,8,9]
a_np = np.array(a_py)

print(a_py[3:7:2], a_np[3:7:2])
print(a_py[2:-1:2], a_np[2:-1:2])
print(a_py[::-1], a_np[::-1])

idx = np.array([7, 2])
a_np[idx]

ages = np.random.randint(low=30, high=60, size=10)
heights = np.random.randint(low=150, high=210, size=10)

print(ages)
print(heights)

print(ages<50)
print(heights[ages<50])
print(ages[ages<50])

shuffled_idx = np.random.permutation(10)
print(shuffled_idx)
print(ages[shuffled_idx])
print(heights[shuffled_idx])

a = np.array([4,5,6])
b = np.array([2,2,2])
a * b

a = np.array([4,5,6])
b = 2
a * b

a = np.arange(10).reshape(1,10)
b = np.arange(12).reshape(12,1)

print (a)
print (b)

print (a * b)

# Matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 7]

x = np.linspace (-2*np.pi, 2*np.pi, 400)
y = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

x = np.linspace(0,2*np.pi,400)
y1= np.tanh(x)
y2 = np.cos(x**2)
fig, ax = plt.subplots(1,2,sharey=True)
ax[1].plot(x,y1)
ax[1].plot(x,-y1)
ax[0].plot(x,y2)
plt.show()