import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# push in four small circle ranges in four positions
# claim they all from seed 102

interf_value_numpy1 = np.load('your interface_dir1')
interf_value_numpy2 = np.load('your interface_dir2')
interf_value_numpy3 = np.load('your interface_dir3')
interf_value_numpy4 = np.load('your interface_dir4')
interf_value_numpy5 = np.load('your interface_dir5')


interf_value_all = np.vstack((interf_value_numpy1, interf_value_numpy2, interf_value_numpy3, interf_value_numpy4, interf_value_numpy5))
print(interf_value_all.shape)

# labels = [1]*5000 + [2]*5000 + [3]*5000 + [4]*5000

labels = ['red']*5000 + ['blue']*5000 + ['green']*5000 + ['purple']*5000

pca1 = PCA(n_components=2)
pca2 = PCA(n_components=2)
pca3 = PCA(n_components=2)
pca4 = PCA(n_components=2)
pca5 = PCA(n_components=2)

pca1.fit(interf_value_numpy1)
pca2.fit(interf_value_numpy2)
pca3.fit(interf_value_numpy3)
pca4.fit(interf_value_numpy4)
pca5.fit(interf_value_numpy5)

interf_value_numpy1 = pca1.transform(interf_value_numpy1)
interf_value_numpy2 = pca2.transform(interf_value_numpy2)
interf_value_numpy3 = pca3.transform(interf_value_numpy3)
interf_value_numpy4 = pca4.transform(interf_value_numpy4)
interf_value_numpy5 = pca5.transform(interf_value_numpy5)

# Select the 0th feature: xs, and 1st feature: ys
xs1 = interf_value_numpy1[:, 0]
ys1 = interf_value_numpy1[:, 1]

xs2 = interf_value_numpy2[:, 0]
ys2 = interf_value_numpy2[:, 1]

xs3 = interf_value_numpy3[:, 0]
ys3 = interf_value_numpy3[:, 1]

xs4 = interf_value_numpy4[:, 0]
ys4 = interf_value_numpy4[:, 1]

xs5 = interf_value_numpy5[:, 0]
ys5 = interf_value_numpy5[:, 1]

pointsize = 0.1

# Scatter plot, coloring by variety_numbers
fig, axs = plt.subplots(3, 2, figsize=(12, 24))
axs[0, 0].scatter(xs1,ys1, s=pointsize, c=labels)
axs[0, 0].set_title('101')
axs[0, 1].scatter(xs2,ys2, s=pointsize, c=labels)
axs[0, 1].set_title('102')
axs[1, 0].scatter(xs3,ys3, s=pointsize, c=labels)
axs[1, 0].set_title('103')
axs[1, 1].scatter(xs4,ys4, s=pointsize, c=labels)
axs[1, 1].set_title('104')
axs[2, 0].scatter(xs5,ys5, s=pointsize, c=labels)
axs[2, 0].set_title('105')

fig.suptitle("Large policy network - Reaching Task\n Ours with relative representation", fontsize=30)
plt.show()
