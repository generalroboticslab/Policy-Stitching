import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
# push in four small circle ranges in four positions

interf_value_numpy1 = np.load('your interface_dir1')
interf_value_numpy2 = np.load('your interface_dir2')
interf_value_numpy3 = np.load('your interface_dir3')
interf_value_numpy4 = np.load('your interface_dir4')

sum_cos_dist12 = 0
sum_cos_dist13 = 0
sum_cos_dist14 = 0
sum_cos_dist23 = 0
sum_cos_dist24 = 0
sum_cos_dist34 = 0

sum_l2_dist12 = 0
sum_l2_dist13 = 0
sum_l2_dist14 = 0
sum_l2_dist23 = 0
sum_l2_dist24 = 0
sum_l2_dist34 = 0

for i in range(len(interf_value_numpy1)):
    sum_cos_dist12 += distance.cosine(interf_value_numpy1[i], interf_value_numpy2[i])
    sum_cos_dist13 += distance.cosine(interf_value_numpy1[i], interf_value_numpy3[i])
    sum_cos_dist14 += distance.cosine(interf_value_numpy1[i], interf_value_numpy4[i])
    sum_cos_dist23 += distance.cosine(interf_value_numpy2[i], interf_value_numpy3[i])
    sum_cos_dist24 += distance.cosine(interf_value_numpy2[i], interf_value_numpy4[i])
    sum_cos_dist34 += distance.cosine(interf_value_numpy3[i], interf_value_numpy4[i])

    sum_l2_dist12 += distance.euclidean(interf_value_numpy1[i], interf_value_numpy2[i])
    sum_l2_dist13 += distance.euclidean(interf_value_numpy1[i], interf_value_numpy3[i])
    sum_l2_dist14 += distance.euclidean(interf_value_numpy1[i], interf_value_numpy4[i])
    sum_l2_dist23 += distance.euclidean(interf_value_numpy2[i], interf_value_numpy3[i])
    sum_l2_dist24 += distance.euclidean(interf_value_numpy2[i], interf_value_numpy4[i])
    sum_l2_dist34 += distance.euclidean(interf_value_numpy3[i], interf_value_numpy4[i])

mean_cos_dist_list = []
mean_cos_dist_list.append(sum_cos_dist12 / len(interf_value_numpy1))
mean_cos_dist_list.append(sum_cos_dist13 / len(interf_value_numpy1))
mean_cos_dist_list.append(sum_cos_dist14 / len(interf_value_numpy1))
mean_cos_dist_list.append(sum_cos_dist23 / len(interf_value_numpy1))
mean_cos_dist_list.append(sum_cos_dist24 / len(interf_value_numpy1))
mean_cos_dist_list.append(sum_cos_dist34 / len(interf_value_numpy1))

mean_cos_dist = np.mean(mean_cos_dist_list)
std_cos_dist = np.std(mean_cos_dist_list)

print("=============================")
print("cosine distance")
print(mean_cos_dist_list)
print(mean_cos_dist)
print(std_cos_dist)

mean_l2_dist_list = []
mean_l2_dist_list.append(sum_l2_dist12 / len(interf_value_numpy1))
mean_l2_dist_list.append(sum_l2_dist13 / len(interf_value_numpy1))
mean_l2_dist_list.append(sum_l2_dist14 / len(interf_value_numpy1))
mean_l2_dist_list.append(sum_l2_dist23 / len(interf_value_numpy1))
mean_l2_dist_list.append(sum_l2_dist24 / len(interf_value_numpy1))
mean_l2_dist_list.append(sum_l2_dist34 / len(interf_value_numpy1))

mean_l2_dist = np.mean(mean_l2_dist_list)
std_l2_dist = np.std(mean_l2_dist_list)

print("=============================")
print("L2 distance")
print(mean_l2_dist_list)
print(mean_l2_dist)
print(std_l2_dist)

cos_map = np.zeros((4, 4))
cos_map[0][1] = mean_cos_dist_list[0]
cos_map[0][2] = mean_cos_dist_list[1]
cos_map[0][3] = mean_cos_dist_list[2]
cos_map[1][2] = mean_cos_dist_list[3]
cos_map[1][3] = mean_cos_dist_list[4]
cos_map[2][3] = mean_cos_dist_list[5]

cos_map[1][0] = cos_map[0][1]
cos_map[2][0] = cos_map[0][2]
cos_map[3][0] = cos_map[0][3]
cos_map[2][1] = cos_map[1][2]
cos_map[3][1] = cos_map[1][3]
cos_map[3][2] = cos_map[2][3]

anot_size = 40
font_size = 40
figure(figsize=(14, 12))
labels = ['train 1', 'train 2', 'train 3', 'train 4']

ax = sns.heatmap(cos_map, cmap='Blues', annot=True, vmin=0, vmax=1, annot_kws={"size": anot_size})
ax.set_xticklabels(labels, fontsize=anot_size)
ax.set_yticklabels(labels, fontsize=anot_size)

plt.title("   Cosine distance of the networks \nwithout relative representation", fontsize=anot_size)
plt.show()

l2_map = np.zeros((4, 4))
l2_map[0][1] = mean_l2_dist_list[0]
l2_map[0][2] = mean_l2_dist_list[1]
l2_map[0][3] = mean_l2_dist_list[2]
l2_map[1][2] = mean_l2_dist_list[3]
l2_map[1][3] = mean_l2_dist_list[4]
l2_map[2][3] = mean_l2_dist_list[5]

l2_map[1][0] = l2_map[0][1]
l2_map[2][0] = l2_map[0][2]
l2_map[3][0] = l2_map[0][3]
l2_map[2][1] = l2_map[1][2]
l2_map[3][1] = l2_map[1][3]
l2_map[3][2] = l2_map[2][3]

figure(figsize=(14, 12))
labels = ['train 1', 'train 2', 'train 3', 'train 4']

ax = sns.heatmap(l2_map, cmap='Blues', annot=True, vmin=0, vmax=10, annot_kws={"size": anot_size})
ax.set_xticklabels(labels, fontsize=anot_size)
ax.set_yticklabels(labels, fontsize=anot_size)

plt.title("   L2 distance of the networks \nwithout relative representation", fontsize=anot_size)
plt.show()

