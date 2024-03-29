<h3> <center>ICX Media Data Science Task</center></h3>

<h5> <center>Samira Kumar Varadharajan</center></h5>

##### Inside the repository:
- `code`: Contains the jupyter notebook [Pre-Process.ipynb](code/Pre-Process.ipynb), 
[main.py](code/main.py) contains the production equivalent code.
- `data`: Contains the dataset for the task.
- `docs`: Contains images for various analysis. `main.py` refers this folder to save plots.
- `model`: Contains the model saved while running `main.py`.  
 
`Pre-Process.ipynb` does a lot of exploratory analysis and tries out different approaches.

`main.py` is the production level code for the task.  

----

##### DATASET:

[SCADI Data Set](https://archive.ics.uci.edu/ml/datasets/SCADI) - UCI ML Repository

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
    class6 = Eating, Drinking, Washing oneself, Caring for body parts, toileting,Dressing, 
    Looking after one's health and Looking after one's safety problem; 
    class7 = No Problem)

----

### <u>Goal:</u>

Based on the available features, some of the questions that can be answered from the project are:
- Would it be possible to cluster children and compare how similar (distance) they're to
each other? 
- Can we identify groups of similar children and provide them similar healthcare/treatment in future?
- Identify children who should belong to a different group than the one they're currently in?
- Identify children who might face potential problems in future.
- Identify the factors that are unique to each cluster or sub-groups. 
- Once clusters are created, can it be effectively used to create target labels for new data?

To answer these questions, first we'd have to find the clusters (unsupervised) and then 
identify the characteristics of each cluster. Then these clusters are considered as target labels
and used in a classification model (supervised) to classify future data. 

----

#### CONTENTS

<u>Clustering:</u>
- [Data cleaning and exploratory analysis:](#EDA)
- [Feature Engineering](#FE)
- [Hierarchical Clustering - Dendrogram](#DE)
- [Visualize clusters using t-SNE](#TSNE)
- [Identify Cluster characteristics](#CC)

<u>Classification:</u>
- [Classification on features vs target label (cluster)](#CLASS)
- [Evaluation of model](#EVAL)
- [Scaling the model to production](#SCALE)
- [Conclusion](#CONCLUDE)

<a name='EDA'/>

#### Data cleaning and exploratory analysis:

- Changed the `Gender` variable to `Categorical` data type. 
- Variable `Age` is changed to `Numerical` (continuous variable) type.
- `Classes` variable is stripped of word 'class' and converted as `Categorical` type. 

```
df['Gender'] = pd.Categorical(df['Gender'])
df['Age'] = pd.to_numeric(df['Age'])
df['Classes'] = df['Classes'].str.replace('class','')
df['Classes'] = pd.Categorical(df['Classes'])
```

#### Exploratory Analysis:

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

Since the data has 206 features, considering all the features for the clustering is not an ideal practice
since around 203 features are sparse (0s and 1s) and having higher feature dimension might not 
provide the best solution for the model. So in order to reduce the feature dimension, we're using 
`Truncated Singular Value Decomposition (TruncatedSVD)` to reduce feature dimension to `50` and
maintain variance of `0.99`. TSVD is used here instead of PCA because input data is sparse and TSVD 
works better on sparse data compared to PCA.


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

**Note:** For this clustering problem, I've included the variable `classes` as a feature 
in order to identify similar children across different classes i.e:- Some children
might have similar features/attributes even though they might be tagged under a different 
class. So in order to identify this pattern, variable `classes` is considered as a feature. 

----

<a name='DE'/>

#### Hierarchical Clustering - Dendrogram:

Once the feature size has been reduced to a better scale, I'm using a `hierarchical 
clustering (Agglomerative)` to find the cluster for each child. 

The main reason for using Hierarchical clustering compared to other options are:

- In this data set, it's ideal to not just cluster children but also find the distance 
or similarity between each child or cluster. This would provide us information as to which 
child is similar to another and maybe drill up/down to nearby cluster groups. 
- The hierarchy helps us to find similar sub-groups within each group, which could
help doctors/ SME's to find different patterns within each group and also find 
interaction of these groups to each other. 
- For example, if 2 children have fever, they might be in same cluster (could be the lowest starting cluster) 
and closer to each other. On the next level, you might have children who have more complicated disease. 
So if these 2 children are closer to the other complicated group, then doctor's could proactively identify 
these 2 children and stop them from going to worse condition.
- From the data set perspective, the data set is small and hence agglomerative clustering
works perfectly and also plotting the dendrogram is easy and can help us identify similar
groups within the data set.

But before starting to create the dendrogram, we'd have to find out the `linkage` 
type and `distance` metrics that are needed for the clustering algorithm. 

Considering the different possible choices of linkage and distance metrics, 
the best choice was found out as below.

For finding the best metric, we're considering the target label to be the `classes`.

```
X = features_sparse_tsvd
y = df.loc[:, 'Classes'].astype(int)
y = y.values
y = y.flatten()
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

So the combination of `Euclidean` and `Ward` provides the best `Adjusted Rand Index (ARI)` (similarity in 
clusters), `Silhouette score` and `Completeness`. Higher the value, closer to 1, better the metrics. 


Using these parameters, we can plot the dendrogram and visualize the agglomerative clustering for the data. 

```
plt.figure(figsize=(15, 15))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(features_sparse_tsvd, method='ward'),orientation='top')
plt.tick_params(axis="x", labelsize=10,rotation='auto')
```

<img src="/docs/dendogram.png" alt="Dendrogram" width="600"/>


From the dendrogram, we can see that the optimal number of clusters are 4 (based on the cut-off rule).

Based on that information, the agglomerative clustering is done as below:

```
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster_pred = cluster.fit_predict(features_sparse_tsvd)
```

----

<a name='TSNE'/>

#### Visualize clusters using t-SNE:

In order to visualize the clusters of 50 features, we use `t-distributed Stochastic Neighbor Embedding (t-SNE)`.
Using t-SNE, the features are reduced to 2 dimensions so that we can visualize the clusters using a scatter plot.


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

<img src="/docs/clusters.png" alt="Clusters" width="600"/>

(Numbers are each patient/child)


----

<a name='CC'/>

#### Identify cluster characteristics:

The characteristics of each cluster is explained below. 

**Average/Median Cluster Age**: 

```
Average age per cluster
Cluster Average_Age
0    11.653846
1    17.500000
2    11.100000
3     8.800000
```

```
Median age per cluster
Cluster  Median_Age
0    12
1    18
2    11
3     9
```

We can see that children in cluster 0 and 2 are similar in age groups and cluster 1 and 3
are extremely opposite (18 and 9 year old respectively). 

```
Total Gender per cluster
Cluster  Gender  Total
0        1         14
         0         12
1        0         10
         1          4
2        0         14
         1          6
3        0          5
         1          5
```

In terms of gender, female children dominate cluster 1 and 2, while male children 
dominate cluster 0. Cluster 3 has equal proportion of both gender. 

Until now, we could conclude that each cluster has:

```
Cluster 0: High male children, aged around 12
Cluster 1: High female children, aged around 18 
Cluster 2: High female children, aged around 12
Cluster 3: Equal proportion of both gender, aged around 8
```

In terms of `classes`, some of the findings between clusters are as below. 

```
Total Classes per cluster
Cluster  Classes  Total
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
is also dominated by class 6 and cluster 3 has class 7 alone. Based on the data description, 
cluster 3 has all children are in ideal condition and have no problem. 
So children with similar characteristics, but away from the cluster 3, 
can be considered that they might soon become `class 7` one day.

```
So cluster 3 has the best possible group of children where the age is lower (around 9) 
and they've no problems. So this cluster acts as the focal point for all clusters where
children in other clusters have problems and would ideally want to move to cluster 3 as 
problem free.
```

Looking at the main features of self care activities, each cluster has these
top 2 characteristics:

```
Cluster 0: Choosing appropriate clothing, avoid risk of abuse of drugs or chemicals
Cluster 1: Carrying out urination appropriately, carrying out defecation properly
Cluster 2: Washing whole body, drying oneself
Cluster 3: Indicating need for urination, carrying out urination appropriately
```

<img src="/docs/top10.png" alt="Top 10 features for each cluster" width="600"/>



**Note:** The self care activities were identified from the below table.

<img src="/docs/categories.png" alt="categories" width="400"/>



We could immediately identify that there are overlapping features between different cluster,
which could indicate their distance / similarity between children. Using a hierarchical
clustering affirms the fact that children who possesses similar features can be easily identified 
and isolated. 

**Overall cluster characteristics:**

Combining all the things that were learnt so far for each cluster, we could say that:

```
Cluster 0: Mostly male children, aged around 12, choosing clothing and avoid drugs
Cluster 1: Mostly female children, aged around 18, with proper sanitation 
Cluster 2: Mostly female children, aged around 12, with proper body wash
Cluster 3: Equal proportion of both gender, aged around 8, with proper sanitation indication
```

<img src="/docs/cluster_c.png" alt="Clusters Explanation" width="600"/>


----

<a name='CLASS'/>

#### Classification on features vs target label (cluster)

Since we've the created an unsupervised clusters/groups of children, 
we can use that as a label to train a classification model which can be used to classify
new patients. The cluster labels are now considered as target labels and the existing features from
TSVD are the features for the model. 

We've a good class balance in this scenario. In case we don't have proper class balance, we can use 
an algorithm's inbuilt class balance penalty factor or use separate methods like `SMOTE` to impute 
synthetic data. 

<img src="/docs/class_balance.png" alt="Class Balance" width="400"/>


In terms of choosing the classification model, `Random Forest` was chosen because of following reasons:

- The data set is small (70 samples), hence using a complicated model like xgboost or neural networks would
not be ideal. 
- Although there is a good class balance, using a bagging method would stop overfitting and would 
perform better than a simple `decision tree`.
- Random Forest works well on high dimension data, handling outliers, mix of categorical and numerical features.


The data was split into 60-40%  for train and testing. 

```
clf=RandomForestClassifier(n_estimators=100, random_state=40)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
```

The initial micro-average of the model was `93 %` but it could be improved by 
using additional methods like `cross validation` for training model 
and `Randomized search` to find best hyper-parameters for the model. 

```
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)

random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }

rfc_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rfc_random.fit(X_train, y_train)

print(rfc_random.best_params_)
```  

Output:

```
Fitting 3 folds for each of 100 candidates, totalling 300 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:   18.5s
[Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:  1.2min
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  2.1min finished
{'n_estimators': 600, 'max_features': 'sqrt', 'max_depth': 420}
```
The new hyper-parameter `{'n_estimators': 600, 'max_features': 'sqrt', 'max_depth': 420}` improved 
the micro-average to `95%`. 

----

<a name='EVAL'/>

#### Evaluation of Model

The best parameters were used to build the Random Forest model. 

```Overall Accuracy: 89%```

In problems like classification, overall accuracy would not explain 
how each class is performing on its own. So to get a better sense of the model, we should look at
`confusion matrix, precision recall and f1-score, ROC curve `.

**Evaluation metrics:** 

<img src="/docs/metrics_classifier.png" alt="Classifier Metrics" width="600"/>



Looking at the `confusion matrix`, class 1 has 2 errors and class 3 has 1 error. This is low but 
the cost of making an error could be different in each cluster/class.

`Type 1 error:`

In a case where a child has bad habits like drugs (cluster 3) but model predicts that child would 
avoid drugs (cluster 0), then the issue is critical since failure to identify this issue would 
have huge impact/cost on the child. 

`Type 2 error:`

For example, class 1 has children with proper sanitation.
So if they're wrongly categorized into a class where children are with improper sanitation, then mistake
is not a big problem since more emphasis would be made on a child to improve sanitation even when their
sanitation is proper. 

Even though the data set is small, we could see that the features have a good 
indication towards the target classes.

Overall, class 0 has lowest `precision` of `0.75` and class 1 and 3 has lowest `recall` of `0.75`. 
Since the classes are balanced to some extent, the average accuracy to be considered 
can be either `micro` or `macro` and both are approximately `0.95` and `0.99` respectively.  

Overall the improved model (after random search CV) performs well for all classes considering the
updated ROC and precision/recall outputs.  

<img src="/docs/valid_score.png" alt="Validation Score" width="400"/>


Considering the small sample size (70), the model's `cross validation score` is lower than 
the `training score` but the difference is small. Also the model suffers from variance since there is a high shaded region around
the cross validation score in below graph. 


----

<a name='SCALE'/>

#### Scaling the model to production

In order to replicate the process in a production environment, few things have to be considered when
developing the process.

The entire process has to be split into 2 parts - Algorithm/Model related activities vs 
Non Algorithm/Model activities. This gives the flexibility to update or modify the pipeline 
based on the type of update and need. Also, it's easy to isolate the issue and fix them and also easy
to add new updates or modify like model parameters.


The outline of the process is sketched below:


<img src="/docs/scaling.png" alt="Scaling" width="600"/>



```
The non-algorithm part (black) includes exploratory graphs, processing data, dendrogram, clusters and 
plotting evaluation metrics. These are essentially the by-products of the algorithm part.

The algorithm/model part (red) does the heavy work including feature reduction, creating clusters,
split data, create, optimize and save models. 

Additionally some components like hyper-parameter tuning has to be done only once for a given set of data.
They have to be updated/rerun only when new data is received.
```


All of these above steps have been implemented in the [main.py](code/main.py) file 
located inside the `code` folder (This file takes approximately 3 minutes to run completely).  

Inside the `main.py`, ```class ClusterClassify``` handles the algorithmic side of the process while 
```class ModelPlots``` handles the non-algorithmic aspect.  


----
<a name='CONCLUDE'/>

**<u>Conclusion:</u>** 

The clusters created provided significant information towards the impact of features
on each cluster and the characteristics of the clusters. The dendrogram shows us which groups of
children are similar (also shows distance between each clusters as well).

Considering the small sample size, the classification model performs efficiently for all 
target classes. The precision and recall for all the classes are high and overall `micro and 
macro average` are high as well. 

Finally, the final saved model (saved from running `main.py`) can be integrated with an
application to predict the cluster/classes for new set of real-time data. 
  