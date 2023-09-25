from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

task_state1 = np.load('path of sampled task states from task1')
task_state2 = np.load('path of sampled task states from task2')
# print(task_state[-20:-10])
task_state_mix = np.concatenate((task_state1, task_state2))
print(task_state_mix.shape)

kmeans = KMeans(n_clusters=128, random_state=0).fit(task_state_mix)
#
center_anchors = kmeans.cluster_centers_
# print(center_anchors)
print(center_anchors.shape)

np.save('kmeans centers path', center_anchors)

center_anchors = np.load('kmeans centers path')

# get the 128 task states which are closest to the 128 kmeans centers
closest, _ = pairwise_distances_argmin_min(center_anchors, task_state_mix)

closest_anchors = np.zeros((128, 15))

for idx in range(len(closest)):
    closest_anchors[idx] = task_state_mix[closest[idx]]

# real task states that are closest to the 128 centroids
np.save('anchors states path', closest_anchors)
