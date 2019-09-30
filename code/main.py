# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.classifier import ClassificationReport, ROCAUC,  ConfusionMatrix

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

import scipy.cluster.hierarchy as shc

import pickle


class ModelPlots:
    """ Functions to plot dendrogram, classifier metrics,
        clusters
    """
    def plot_dendrogram(self):

        plt.figure(figsize=(15, 15))
        plt.title("Patient Dendograms")
        dend = shc.dendrogram(shc.linkage(self.features_tsvd, method='ward'), orientation='top')
        plt.tick_params(axis="x", labelsize=10, rotation='auto')
        plt.savefig('../docs/dendogram.png')
        plt.show()

    def plot_classifier_metrics(self):

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        visualgrid = [
            ConfusionMatrix(self.clf, ax=axes[0][0]),
            ClassificationReport(self.clf, ax=axes[0][1]),
            ROCAUC(self.clf, ax=axes[1][0]),
        ]
        fig.delaxes(axes[1, 1])
        for viz in visualgrid:
            viz.fit(self.X_train, self.y_train)
            viz.score(self.X_test, self.y_test)
            viz.finalize()
        plt.savefig('../docs/metrics_classifier.png')
        plt.show()

    def plot_clusters(self):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="cluster",
            palette=sns.color_palette("hls", 4),
            data=self.features_sparse_tsvd_df_,
            legend="full",
            alpha=1,
            s=100
        )
        for line in range(0, self.features_sparse_tsvd_df_.shape[0]):
            plt.text(self.features_sparse_tsvd_df_["tsne-2d-one"][line] + .3,
                     self.features_sparse_tsvd_df_["tsne-2d-two"][line],
                     self.features_sparse_tsvd_df_.index.values[line], horizontalalignment='left',
                     size='large', color='black')

        plt.savefig('../docs/clusters.png')
        plt.show()


class ClusterClassify(ModelPlots):
    """
    Functions to get feature reduction,
    create clusters, split train and test data,
    get best model parameters, get accuracy,
    save model
    """
    def __init__(self, features_tsvd=None, clf=None):
        self.features_tsvd = None
        self.clf = RandomForestClassifier(n_estimators=600, max_depth=420, max_features='sqrt', random_state=40)

    def get_feature_reduce(self, df):
        tsvd = TruncatedSVD(n_components=50, random_state=40)

        # Conduct TSVD on sparse matrix
        features_sparse_tsvd = tsvd.fit(df).transform(df)


        print("Original number of features:", df.shape[1])
        print("Reduced number of features:", features_sparse_tsvd.shape[1])
        print("Total Variance captured:", tsvd.explained_variance_ratio_.sum())

        self.features_tsvd = features_sparse_tsvd

    def get_clusters(self):
        cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
        cluster_pred = cluster.fit_predict(self.features_tsvd)

        features_sparse_tsvd_df = pd.DataFrame(self.features_tsvd)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=100)
        tsne_results = tsne.fit_transform(features_sparse_tsvd_df)
        features_sparse_tsvd_df['tsne-2d-one'] = tsne_results[:, 0]
        features_sparse_tsvd_df['tsne-2d-two'] = tsne_results[:, 1]

        features_sparse_tsvd_df['cluster'] = cluster_pred
        self.features_sparse_tsvd_df_ = features_sparse_tsvd_df

        df['Cluster'] = cluster_pred

        # Avg age per cluster
        print("\nAverage age per cluster\n", df.groupby('Cluster')['Age'].mean())
        # Gender 1-male, 0-female
        print("\nGender per cluster\n", df.groupby('Cluster')['Gender'].value_counts())
        print("\nClasses per cluster\n", df.groupby('Cluster')['Classes'].value_counts())

        self.df = df

    def get_test_train(self):
        X, y = self.features_tsvd, self.df['Cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

        print('\nTraining Features Shape:', X_train.shape)
        print('Training Labels Shape:', y_train.shape)
        print('Testing Features Shape:', X_test.shape)
        print('Testing Labels Shape:', y_test.shape)

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def get_best_params(self):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']

        max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
        max_depth.append(None)

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth
        }

        rfc_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid,
                                        n_iter=100, cv=3, verbose=1,
                                        random_state=40, n_jobs=-1)

        rfc_random.fit(self.X_train, self.y_train)

        self.clf = rfc_random.best_estimator_

    def get_model_accuracy(self):
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)
        print("\nModel Accuracy %:", np.round(metrics.accuracy_score(self.y_test, y_pred) * 100, 2))

    def get_model_saved(self):
        filename = '../model/finalized_model.sav'
        pickle.dump(self.clf, open(filename, 'wb'))
        print('\nModel saved in model folder in main directory')

    def run_phases(self):
        phases = (self.plot_dendrogram(), self.get_clusters(),
                  self.plot_clusters(), self.get_test_train(),
                  self.get_best_params(), self.plot_classifier_metrics(),
                  self.get_model_accuracy(), self.get_model_saved())
        for phase in phases:
            phase


if __name__ == '__main__':

    df = pd.read_csv("../data/SCADI.csv")
    df['Gender'] = pd.Categorical(df['Gender'])
    df['Age'] = pd.to_numeric(df['Age'])
    df['Classes'] = df['Classes'].str.replace('class', '')
    df['Classes'] = pd.Categorical(df['Classes'])

    df_tsvd = df.copy()
    clf = ClusterClassify()
    clf.get_feature_reduce(df_tsvd)
    clf.run_phases()

