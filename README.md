# sorted_cluster
Sort scikit-learn cluster_centers_

## example

```python
from sorted_cluster import sorted_cluster
import numpy as np
arr = np.vstack([
    100 + np.random.random((2,3)),
    np.random.random((2,3)),
    5 + np.random.random((3,3)),
    10 + np.random.random((2,3))
])
print('Data:')
print(arr)

cluster = KMeans(n_clusters=4)

print('\n Without sort:')
cluster.fit(arr)
print(cluster.cluster_centers_)
print(cluster.labels_)
print(cluster.predict([[5,5,5],[1,1,1]]))

print('\n With sort:')
cluster = sorted_cluster(arr, cluster)
print(cluster.cluster_centers_)
print(cluster.labels_)
print(cluster.predict([[5,5,5],[1,1,1]]))
```

result:
```
Data:
[[100.52656263 100.57376566 100.63087757]
 [100.70144046 100.94095196 100.57095386]
 [  0.21284187   0.75623797   0.77349013]
 [  0.28241023   0.89878796   0.27965047]
 [  5.14328748   5.37025887   5.26064209]
 [  5.21030632   5.09597417   5.29507699]
 [  5.81531591   5.11629056   5.78542656]
 [ 10.25686526  10.64181304  10.45651994]
 [ 10.14153211  10.28765705  10.20653228]]

 Without sort:
[[ 10.19919868  10.46473505  10.33152611]
 [100.61400155 100.75735881 100.60091572]
 [  0.24762605   0.82751296   0.5265703 ]
 [  5.38963657   5.19417453   5.44704855]]
[1 1 2 2 3 3 3 0 0]
[3 2]

 With sort:
[[  0.24762605   0.82751296   0.5265703 ]
 [  5.38963657   5.19417453   5.44704855]
 [ 10.19919868  10.46473505  10.33152611]
 [100.61400155 100.75735881 100.60091572]]
[3 3 0 0 1 1 1 2 2]
[1 0]
```
