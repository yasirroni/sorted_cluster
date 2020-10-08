# sorted_cluster
Sort scikit-learn cluster_centers_

## example

```python
import numpy as np
arr = np.vstack([
    100 + np.random.random((2,3)),
    np.random.random((2,3)),
    5 + np.random.random((3,3)),
    10 + np.random.random((2,3))
])
print(arr)

cluster = KMeans(n_clusters=4)
cluster.fit(arr)
print(cluster.cluster_centers_)
print(cluster.labels_)
print(cluster.predict([[5,5,5],[1,1,1]]))

cluster = sorted_cluster(arr, cluster)
print(cluster.cluster_centers_)
print(cluster.labels_)
print(cluster.predict([[5,5,5],[1,1,1]]))
```
