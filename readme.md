### <center>iCX Media Data Science Task</center>
##### <center>Samira Kumar Varadharajan</center>

##### DATASET:

[SCADI Data Set](https://archive.ics.uci.edu/ml/datasets/SCADI)

**<u>Description:</u>** 

This dataset contains 206 attributes of 70 children with physical and motor disability based on ICF-CY.
In particular, the SCADI dataset is the only one that has been used by ML researchers for self-care problems classification based on ICF-CY to this date.
The 'Class' field refers to the presence of the self-care problems of the children with physical and motor disabilities.The classes are determined by occupational therapists.


**<u>Attribute Information:</u>**

    1: gender: gender (1 = male; 0 = female)
    2: age: age in years
    3-205: self-care activities based on ICF-CY (1 = The case has this feature; 0 = otherwise)
    206: Classes ( class1 = Caring for body parts problem; class2 = Toileting problem; class3 = Dressing problem; 
    class4 = Washing oneself and Caring for body parts and Dressing problem; 
    class5 = Washing oneself, Caring for body parts, Toileting, and Dressing problem; 
    class6 = Eating, Drinking, Washing oneself, Caring for body parts, toileting,Dressing, Looking after oneâ€™s health and Looking after oneâ€™s safety problem; 
    class7 = No Problem)

----

### <u>Goal:</u>

Based on the features, some of the questions that can be answered from the project are:
- Would it be possible to cluster children and compare how similar they're to
each other? 
- Can we identify groups of similar children and provide them similar healthcare?
- Identify children who might belong to a different group than the one they're currently in?
- Identify potential children who might face different problems in future.

----

#### CONTENTS

<u>Clustering:</u>
- [Data cleaning and exploratory analysis:](#EDA)
- [Feature Engineering](#FE)
- [Clustering - Dendrogram (find no of clusters)](#DE)
    - Hierarchical clustering
- [Visualize clusters using t-sne](#TSNE)
- [Cluster characteristics](#CC)

<u>Classification:</u>
- [Classification on features vs target label (cluster)](#CLASS)
- [Evaluation of model](#EVAL)

----

<a name='EDA'/>

#### Data cleaning and exploratory analysis:

- Changing the `Gender` variable to `Categorical`
- `Age` is set to `Numerical` (continuous variable)
- `Classes` variable is stripped of word 'class' and converted as `Categorical`

```df['Gender'] = pd.Categorical(df['Gender'])
df['Age'] = pd.to_numeric(df['Age'])
df['Classes'] = df['Classes'].str.replace('class','')
df['Classes'] = pd.Categorical(df['Classes'])
```

**<u>Gender Distribution</u>**
```
plt.figure(figsize=(7, 4))
plt.title("Gender Distribution")
df['Gender'].value_counts().plot(kind='bar')
plt.show()
```
<img src="/docs/gender_distribution.png" alt="Gender Distribution" width="400"/>


**<u>Age Histogram</u>**

- Large portion of 70 children lie between ages 8 to 16

```
plt.figure(figsize=(7, 4))
plt.title("Age Histogram")
df['Age'].hist(bins=10, alpha=0.9)
plt.show()
```
<img src="/docs/age_histogram.png" alt="Age Histogram" width="400"/>


**<u>Class Distribution</u>**
```
plt.figure(figsize=(7, 4))
plt.title("Classes Distribution")
df['Classes'].value_counts().plot(kind='bar')
plt.show()
```
<img src="/docs/class_distribution.png" alt="Classes Distribution" width="400"/>

----

<a name='FE'/>

#### Feature Engineering:

Since the data has 206 features, considering all the features for the clustering is not a good practice
since around 203 features are sparse (0s and 1s) and having higher features might not 
provide best solution. So in order to reduce the feature dimension, we're using 
`Truncated Singular Value Decomposition (TruncatedSVD)` to reduce features to `50` and also
maintain variance of `0.99`. 

```
df_tsvd = df.copy()
tsvd = TruncatedSVD(n_components=50,random_state=40)

# Conduct TSVD on sparse matrix
features_sparse_tsvd = tsvd.fit(df_tsvd).transform(df_tsvd)

# Show results
print("Original number of features:", df_tsvd.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1]) 
print("Total Variance", tsvd.explained_variance_ratio_.sum())
```

**Output:**

```
Original number of features: 206
Reduced number of features: 50
Total Variance 0.997368053105751
```

For this clustering, I've included the `classes` feature as well in order to identify
similar children across different classes i.e:- Some children might have similar 
features/attributes even though they might be tagged under a different class. So
in order to identify that pattern, I've included `classes` in the features. 

----

<a name='DE'/>

#### Clustering - Dendrogram (find no of clusters):

So once the feature size has been reduced to a better scale, I'm using a hierarchical 
clustering (Agglomerative) to find the cluster for each child. 

The main reason for using Hierarchical clustering compared to other options are:
- In this data set, it's ideal to not just cluster children but also find the distance 
or similarity between each child or cluster. This would provide us information as to which 
child is similar to another and maybe drill up/down to next cluster group. 
- The hierarchy helps us to find similar sub-groups within each group, which could
maybe help doctors/ SME's to find different patterns within each group and also find 
interaction of these groups. 
- For example, if 2 children have fever, they might be in same cluster and closer. 
On the next level, you might have children who have more complicated disease. So if these
2 children are closer to the other group, then doctor's could proactively identify these
2 children and stop them from becoming worse.  
- From the dataset perspective, the dataset is small and hence agglomerative clustering
works perfectly and also plotting the dendrogram can easily help us identify similar
groups within the dataset.

But before starting to create the dendrogram, we'd have to find out the `linkage` 
type and `distance` metrics. 

Considering the possible choices of linkage and distance metrics, 
the best choice was found out as below.

For finding the best metric, we're considering the target label to be the `classes`.

```
X = features_sparse_tsvd
y = df.loc[:, 'Classes'].astype(int)
y = y.values
print(y.shape)
y=y.flatten()
```

```
distances=['cityblock','euclidean']
a=[]
for dist in distances:
    a.append(pdist(X, dist))
linkage = [single, complete, average, weighted, centroid, median, ward]
data = []
for l in linkage:

    clusters=(fcluster(l(a[1]), t=7, criterion='maxclust'))

    data.append(({
        'ARI': metrics.adjusted_rand_score(y, clusters),
        'Completeness': metrics.completeness_score(y, clusters),
        'Silhouette': metrics.silhouette_score(X, clusters)}))
results = pd.DataFrame(data=data, columns=['ARI',
'Completeness', 'Silhouette'],
index=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'])
print(results)
```
**Output:**
```
              ARI  Completeness  Silhouette
single    0.208778      0.634543   -0.042088
complete  0.257458      0.453664    0.136276
average   0.681840      0.703915    0.233113
weighted  0.348448      0.550208    0.141142
centroid  0.175903      0.625429    0.111320
median    0.220518      0.575907    0.075075
ward      0.729150      0.736630    0.230340
```

So the combination of `Euclidean` and `Ward` provides the best `Adjusted Rand Index (ARI)` (Similarity in 
clusters), `Silhouette score`. 


Then we can plot the dendrogram and visualize the agglomerative clustering of the data. 

```
plt.figure(figsize=(15, 15))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(features_sparse_tsvd, method='ward'),orientation='top')
plt.tick_params(axis="x", labelsize=10,rotation='auto')
```

<img src="/docs/dendogram.png" alt="Dendrogram" width="400"/>


From the dendrogram, we can see that the optimal number of clusters are 4. 

The agglomerative clustering is done as below:

```
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster_pred = cluster.fit_predict(features_sparse_tsvd)
```

----

<a name='TSNE'/>

#### Visualize clusters using t-sne:

In order to visualize the clusters of 50 features, we use `t-distributed Stochastic Neighbor Embedding (t-SNE)`.
The features are reduced to 2 dimension so that we can visualize the clusters using a simple scatter plot.


```
features_sparse_tsvd_df = pd.DataFrame(features_sparse_tsvd)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=100)
tsne_results = tsne.fit_transform(features_sparse_tsvd_df)
features_sparse_tsvd_df['tsne-2d-one'] = tsne_results[:,0]
features_sparse_tsvd_df['tsne-2d-two'] = tsne_results[:,1]


features_sparse_tsvd_df['cluster'] = cluster_pred
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="cluster",
    palette=sns.color_palette("hls", 4),
    data=features_sparse_tsvd_df,
    legend="full",
    alpha=1,
    s=100
)
for line in range(0,features_sparse_tsvd_df.shape[0]):
     plt.text(features_sparse_tsvd_df["tsne-2d-one"][line]+.3, features_sparse_tsvd_df["tsne-2d-two"][line], 
              features_sparse_tsvd_df.index.values[line], horizontalalignment='left', 
              size='large', color='black')
```

**Output:**

<img src="/docs/clusters.png" alt="Clusters" width="400"/>

----

<a name='CC'/>

#### Cluster characteristics

**Average/Median Cluster Age**: 

```Average age per cluster
 Cluster
0    11.653846
1    17.500000
2    11.100000
3     8.800000
```

```
Median age per cluster
 Cluster
0    12
1    18
2    11
3     9
```

We can see that children in cluster 0 and 2 are similar in age groups and cluster 1 and 3
are extreme (18 and 9 year old respectively). 

```
Gender per cluster
 Cluster  Gender
0        1         14
         0         12
1        0         10
         1          4
2        0         14
         1          6
3        0          5
         1          5
```

In terms of gender, the female children dominate cluster 1 and 2, while male children 
dominate cluster 0. Cluster 3 has equal proportion. 
Until now, we could say that:

```
Cluster 0: Mostly male children, aged around 12
Cluster 1: Mostly female children, aged around 18 
Cluster 2: Mostly female children, aged around 12
Cluster 3: Equal proportion of children, aged around 8
```

In terms of classes, some of the finding between clusters are as below. 

```
Classes per cluster
 Cluster  Classes
0        4          11
         7           5
         2           4
         1           2
         5           2
         3           1
         6           1
1        6           9
         2           3
         4           1
         7           1
2        6          19
         5           1
3        7          10
```

Cluster 0 is dominated by class 4, cluster 1 is dominated by class 6, cluster 2 
is also dominated by class 6 and cluster 3 has class 7 alone. 

```
So cluster 3 has children in lower age groups having the highest class of problems,
which is understandable.
```

Looking at the main features - self care activities, each cluster has these
 top characteristics:

```
Cluster 0: Choosing appropriate clothing, avoid risk of abuse of drugs or chemicals
Cluster 1: Carrying out urination appropriately, carrying out defecation properly
Cluster 2: Washing whole body, drying oneself
Cluster 3: Indicating need for urination, carrying out urination appropriately
```

<img src="/docs/top10.png" alt="Top 10 features for each cluster" width="400"/>



**Note:** The self care activities were identifed from the below table.

<img src="/docs/categories.png" alt="categories" width="400"/>



We could immediately identify that there are overlapping features between different cluster,
which could indicate their distance / similarity between children. Again using a hierarchical
clustering affirms the fact that a child who possesses similar features can be easily identified 
and isolated amongst a subgroup. 


**<u>Overall Cluster characteristics:</u>**
```
Cluster 0: Mostly male children, aged around 12, choosing clothing and avoid drugs
Cluster 1: Mostly female children, aged around 18, with proper sanitation 
Cluster 2: Mostly female children, aged around 12, with proper body wash
Cluster 3: Equal proportion of children, aged around 8, with proper sanitation indication
```

----

<a name='CLASS'/>

#### Classification on features vs target label (cluster)

Since we've the clusters/groups of children, we can use that as a label to classify future children.

In this case, we've a good class balance:

<img src="/docs/class_balance.png" alt="Class Balance" width="400"/>


In terms of choosing the classification model, I chose `Random Forest` because of following reasons:
- The dataset is small (70 samples), hence using a complicated model like xgboost or neural networks would
not be ideal.
- Although there is a good class balance, using a bagging method would stop overfitting.
- Working on high dimension data, handling outliers, mix of categorical and numerical data.


The data was split into 60-40%  for train and testing. 

```
clf=RandomForestClassifier(n_estimators=100, random_state=40)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
```

----

<a name='EVAL'/>

#### Evaluation of Model

```Overall Accuracy: 85%```

**Evaluation metrics:** 

<img src="/docs/metrics_classifier.png" alt="Classifier Metrics" width="600"/>

Looking at the `confusion matrix`, class 1 has 3 errors and class 3 has 1 error. The cost of making
a mistake could be different in each cluster/class.

`Type 1 error:`

In a case where a child has bad habits like drugs (cluster 3) but model predicts that child would 
avoid drugs (cluster 0), then the issue is critical since failure to identify this issue would 
have huge impact/cost on the child. 

`Type 2 error:`

For example, class 1 has children with proper sanitaion.
So if they're wrongly categorized into a class where children are with improper sanitation, then mistake
is not a big problem since more emphasis would be made on a child to improve sanitation even when their
sanitation is proper. 


Even though the data set is small, we could see that the features have a good 
indication towards the target classes.

Class 0 has lowest `precision` of `0.75` and class 1 has lowest `recall` of `0.625`. 
Since the classes are balanced to some extent, the average accuracy to be considered 
can be either `micro` or `macro` and both are approximately `0.93` and `0.96` respectively.  

<a name='SCALE'/>

----

#### Scaling the model

In order to replicate the model in a production environment, few things have to be considered when
developing the code/process.
The entire process has to be split into 2 parts - Algorithm/Model related activities vs 
Non Algorithm/Model activities. This gives the flexibility to update or modify the pipeline 
based on the type and need.

 <img src="/docs/scaling.png" alt="Scaling" width="600"/>



```
The non-algorithm part includes exploratory graphs, processing data, dendrogram, clusters and 
plotting evaluation metrics. These are essentially the by-products of the algorithm part.

The algorithm/model part does the heavy work including feature reduction, creating clusters,
split data, create and optimize models and saving models. 
```

All of these above methods have been implemented in the [main.py](code/main.py) file inside the `code` folder. 

```class ClusterClassify``` handles the algorithmic side of the process while 
```class ModelPlots``` handles the non-algorithmic aspect.  


**<u>Conclusion:</u>** 

Considering the small sample size, the model performs relatively efficient for all 
target classes. 