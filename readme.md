### <center>iCX Media Data Science Task</center>
##### <center>Samira Kumar Varadharajan</center>

##### DATASET:

[SCADI Data Set](https://archive.ics.uci.edu/ml/datasets/SCADI)

<u>Description:</u> 

This dataset contains 206 attributes of 70 children with physical and motor disability based on ICF-CY.
In particular, the SCADI dataset is the only one that has been used by ML researchers for self-care problems classification based on ICF-CY to this date.
The 'Class' field refers to the presence of the self-care problems of the children with physical and motor disabilities.The classes are determined by occupational therapists.


<u>Attribute Information:</u>

    1: gender: gender (1 = male; 0 = female)
    2: age: age in years
    3-205: self-care activities based on ICF-CY (1 = The case has this feature; 0 = otherwise)
    206: Classes ( class1 = Caring for body parts problem; class2 = Toileting problem; class3 = Dressing problem; 
    class4 = Washing oneself and Caring for body parts and Dressing problem; 
    class5 = Washing oneself, Caring for body parts, Toileting, and Dressing problem; 
    class6 = Eating, Drinking, Washing oneself, Caring for body parts, toileting,Dressing, Looking after oneâ€™s health and Looking after oneâ€™s safety problem; 
    class7 = No Problem)

----

#### CONTENTS

<u>Clustering:</u>
- [Data cleaning and exploratory analysis:](#EDA)
- Feature Engineering
- Clustering - Dendrogram (find no of clusters)
    - Hierarchical clustering
- Visualize clusters using t-sne
- Identify cluster features

<u>Classification:</u>
- Classification on features vs target label (cluster)
- Evaluation of model
- Save model for future use

----

<a name='EDA'/>

#### Data cleaning and exploratory analysis:

- Changing the `Gender` variable to `Categorical`
- `Age` is set to numerical (continuous variable)
- `Classes` variable is stripped of word 'class' and converted as `Categorical`

```df['Gender'] = pd.Categorical(df['Gender'])
df['Age'] = pd.to_numeric(df['Age'])
df['Classes'] = df['Classes'].str.replace('class','')
df['Classes'] = pd.Categorical(df['Classes'])
```

<img src="/docs/gender_distribution.png" alt="Soccer Animation" width="700"/>

![Gender Distribution](..docs/gender_distribution.png)



