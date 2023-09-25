import numpy as np
from matplotlib import pyplot as plt

# push in four small circle ranges in four positions

interf_value_numpy1 = np.load('your interface_dir1')
interf_value_numpy2 = np.load('your interface_dir2')
interf_value_numpy3 = np.load('your interface_dir3')
interf_value_numpy4 = np.load('your interface_dir4')

interf_value_all = np.vstack((interf_value_numpy1, interf_value_numpy2, interf_value_numpy3, interf_value_numpy4))
print(interf_value_all.shape)

labels = ['red']*5000 + ['blue']*5000 + ['green']*5000 + ['purple']*5000

# Select the 0th feature: xs, and 1st feature: ys
xs1 = interf_value_numpy1[:, 0]
ys1 = interf_value_numpy1[:, 1]
zs1 = interf_value_numpy1[:, 2]

xs2 = interf_value_numpy2[:, 0]
ys2 = interf_value_numpy2[:, 1]
zs2 = interf_value_numpy2[:, 2]

xs3 = interf_value_numpy3[:, 0]
ys3 = interf_value_numpy3[:, 1]
zs3 = interf_value_numpy3[:, 2]

xs4 = interf_value_numpy4[:, 0]
ys4 = interf_value_numpy4[:, 1]
zs4 = interf_value_numpy4[:, 2]

fig = plt.figure(figsize=(12, 15))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(xs1, ys1, zs1, s=30.0, c=labels)
ax.set_title('101')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(xs2, ys2, zs2, s=30.0, c=labels)
ax.set_title('102')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(xs3, ys3, zs3, s=30.0, c=labels)
ax.set_title('103')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.scatter(xs4, ys4, zs4, s=30.0, c=labels)
ax.set_title('104')

fig.suptitle("Small policy network - Reaching Task\n Ours with relative representation", fontsize=30)
plt.show()
