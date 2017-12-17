import matplotlib
import numpy as np
import seaborn as sb

orb_data = [0.58, 0.10, 0, .43]
orb_average = 0.325015662079

sift_data = [.51, .10, .33, .49]
sift_average = 0.383317206158

data = [orb_data, sift_data]
arr = np.asarray(data)
arr = arr.transpose()
print(arr)

labels = ['Shanghai', 'London', 'New York City', 'Chicago']

sb.set(style="white")

ax = sb.barplot(data=arr)
matplotlib.pyplot.show()
