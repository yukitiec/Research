# by typing Header, writing snippet automatically
# @file      :    InternIris_default.py
#
# @brief     :     None
#
# @author    :     Yuki Kawawaki
# @contact   :     kawawaki-yuki628@g.ecc.u-tokyo.ac.jp
# @data      :     2023/02/17
# (C)Copyright 2023, WALC Inc.

# import library
import csv
import statistics
import warnings

import graphviz
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
#model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#data transformation
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
#PCA (Principal Component Analysis)
from sklearn.decomposition import PCA
#NMF (Non-Negative Matricx Factorization)
from sklearn.decomposition import NMF
#TSNE : (manifold learning algorithms)
from sklearn.manifold import TSNE
#K-Means Method
from sklearn.cluster import KMeans
#dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
#determine the num of clusters
from scipy.cluster.hierarchy import inconsistent
#DBSCAN
from sklearn.cluster import DBSCAN

warnings.filterwarnings(
    "ignore", category=DeprecationWarning)  # Ignore warinings


class AnalyzeIris:
    # TODO: Add docstring

    def __init__(self, n_neighbors: int):
        """initialize parameters

        Args:
            n_neighbors (int): Nearest Neighbors in K-Neighbors Classifier
        """
        # Data
        self.iris = load_iris()
        # iris data for PairPlot
        self.df = self.Get()
        # Iris, Supervised train
        # model define
        self.logreg = LogisticRegression()
        self.linsvc = LinearSVC()
        self.tree = DecisionTreeClassifier()  # random_state = 0)
        self.lr = LinearRegression()
        self.n_neighbors = n_neighbors
        self.knc = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.forest = RandomForestClassifier(random_state=0)
        self.grbt = GradientBoostingClassifier(
            random_state=0)  # ,learning_rate=0.01)
        self.mlp = MLPClassifier(random_state=0)
        self.model = [self.logreg, self.linsvc, self.tree,
                      self.lr, self.knc, self.forest, self.grbt, self.mlp]
        # model name
        self.model_name = ["LogisticRegression", "LinearSVC", "DecisionTreeClassifier", "LinearRegression", "KNeighborsClassifier",
                           "RandomForestClassifier", "GradientBoostingClassifier", "MLPClassifier"]
        # Setting : Train Condition
        # stratified cross-validation : split = 5
        self.kf = StratifiedKFold(n_splits=5, shuffle=True)
        # Train: test_result, trained model
        self.test_result, self.model_trained = self.Calculate_result()
        # test data as dataframe
        self.result_dataframe = self.GetSupervised()

    def Get(self) -> list:
        """Show Iris data

        Returns:
            list: Iris data(150) ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','label']
        """

        # Load data from sklearn.datasets
        iris =self.iris
        file_iris = './iris_data.csv'
        iris_data = iris["data"]
        iris_label = iris["target"]
        num_data = iris_label.shape[0]

        # Write necessary Iris data in csv file
        with open(file_iris, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['sepal length (cm)', 'sepal width (cm)',
                            'petal length (cm)', 'petal width (cm)', 'Label'])
        for i in range(num_data):
            with open(file_iris, "a") as f:
                writer = csv.writer(f)
                writer.writerow([iris_data[i, 0], iris_data[i, 1],
                                iris_data[i, 2], iris_data[i, 3], iris_label[i]])

        # Load csv file and show
        df = pd.read_csv(filepath_or_buffer=file_iris)
        return df

    def PairPlot(self, cmap: str = mglearn.cm3) -> None:
        """Plotting Scatter matrix of Iris data

        Args:
            cmap (str): color condition
        """
        # get Iris data as dataframe
        df = self.df
        # get feature name
        feature_name = df.columns[:4]
        # convert to nd.array in order to split data :shape(150,5)
        iris_value = df.values
        # feature data
        feature = iris_value[:, :4]
        # label data
        label = iris_value[:, -1]

        # convert to DataFrame with features and feature name
        iris_dataframe = pd.DataFrame(feature, columns=feature_name)

        # plot scatter_matrix
        pd.plotting.scatter_matrix(iris_dataframe, c=label, figsize=(15, 15), marker='o',
                                   hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=cmap)

    def Show_result(self, classifier: object, name: str) -> object:
        """show clssification results 

        Args:
            classifier (object): Classify model
            name (str): name of classify model

        Returns:
            object: trained model
        """
        # load data
        feature, label = self.iris.data, self.iris.target
        kf = self.kf
        # show data
        print("=== {} ===".format(name))
        for train_index, test_index in kf.split(feature, label):
            train_data, train_label, test_data, test_label = feature[
                train_index], label[train_index], feature[test_index], label[test_index]
            classifier.fit(train_data, train_label)
            result_train = classifier.score(train_data, train_label)
            result_test = classifier.score(test_data, test_label)
            print("test score: {:.3f}   train score: {:.3f}".format(
                result_test, result_train))
        return classifier

    def AllSupervised(self, n_neighbors: int = 4) -> None:
        """show classification result of 8 classifier

        Args:
            n_neighbors (int, optional): KNearestNeighbors num of candidates  Defaults to 4.

        """
        # load iris data
        iris = self.iris
        feature, label = iris.data, iris.target
        # model load
        model = self.model
        model_name = self.model_name
        for i in range(len(model_name)):
            classifier = self.Show_result(model[i], model_name[i])

    def Calculate_result(self, n_neighbors: int = 4) -> (list, list):
        """calculate the classificaton result of 8 model

        Args:
            n_neighbors (int, optional): KNearestNeighbors num of candidates  Defaults to 4.

        Returns:
            list : test result
            list: pretrained model
        """
        # load iris data
        iris = self.iris
        feature, label = self.iris.data, self.iris.target
        # model load
        model = self.model
        model_name = self.model_name
        # stratified cross-validation : split = 5
        kf = self.kf
        test_result = []
        for train_index, test_index in kf.split(feature, label):
            temp_result = []
            # keep trained models : revise every train dataset
            model_trained = []
            for classifier in model:
                train_data, train_label, test_data, test_label = feature[
                    train_index], label[train_index], feature[test_index], label[test_index]
                classifier.fit(train_data, train_label)
                model_trained.append(classifier)
                result_test = classifier.score(test_data, test_label)
                temp_result.append(round(result_test, 3))
            test_result.append(temp_result)
        return test_result, model_trained

    def GetSupervised(self) -> list:
        """show classification result data

        Returns:
            list: result data of classification
        """
        # get results of classification
        test_result = self.test_result
        model_name = self.model_name
        # convert data into dataframe
        result_dataframe = pd.DataFrame(test_result, columns=model_name)
        return result_dataframe

    def BestSupervised(self):
        """show best model and the score of the mdoel

        Args:
            self (_type_): _description_
            float (_type_): _description_

        Returns:
            str :  best model name
            float : best mean score of the model
        """
        # get result of classification as a dataframe
        result_dataframe = self.result_dataframe
        # convert data into numpy array
        result_array = result_dataframe.values
        # list for mean result
        mean_result = []
        # calculate mean for classification accuracy
        num_model = result_array.shape[1]
        for i in range(num_model):
            mean_result.append(statistics.mean(result_array[:, i]))
        # best model index
        best_index = np.argmax(mean_result)
        best_model = result_dataframe.columns[best_index]
        best_score = mean_result[best_index]
        return best_model, best_score

    def plot_importances_iris(self, model: object, model_name: str) -> None:
        """show importances of each features

        Args:
            model (object): trained model
            model_name (str): model name
        """
        name_features = self.iris.feature_names
        num_features = len(self.iris.feature_names)
        plt.barh(range(num_features),
                 model.feature_importances_, align="center")
        plt.yticks(np.arange(num_features), name_features)
        plt.xlabel("Feature importances : {}".format(model_name))
        plt.ylabel("Feature")
        plt.show()

    def PlotFeatureImportancesAll(self) -> None:
        """show importances of iris features
            DecisionTree, RandomForest, GradientBoosting
        """
        # get trained model
        model_trained = self.model_trained
        print(len(model_trained))
        tree = self.model_trained[2]
        forest = self.model_trained[5]
        grbt = self.model_trained[6]
        models = [tree, forest, grbt]
        # get name of the model
        name_tree = self.model_name[2]
        name_forest = self.model_name[5]
        name_grbt = self.model_name[6]
        model_names = [name_tree, name_forest, name_grbt]
        for model, name in zip(models, model_names):
            self.plot_importances_iris(model, name)

    def VisualizeDecisionTree(self) -> None:
        """Show Graph of Decision Tree
        """
        model_trained = self.model_trained
        # get trained model
        tree = self.model_trained[2]
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        sklearn.tree.plot_tree(tree,
                               feature_names=self.iris.feature_names,
                               class_names=["L", "a", "b"],
                               filled=True)
        '''
        #write file as .dot
        export_graphviz(tree,out_file ="./tree.dot",class_names = ["b","a","L"],
                       feature_names = self.iris.feature_names,impurity=False,filled = True)
        with open("./tree.dot") as f:
            dot_graph = f.read()
        graphviz.Source(dot_graph)
        #visualize_tree = mglearn.plots.plot_tree_not_monotone()
        #diplay(visualize_tree)
        '''

    def PlotScaledData(self) -> None:
        """Compare the influence of scaling in LinearSVC by showing results and plotting data
        """
        # load iris data
        iris = self.iris
        features, labels, feature_names = iris.data, iris.target, iris.feature_names
        # load LinearSVC model
        linsvc = self.linsvc
        # stratified cross-validation : split = 5
        kf = self.kf
        # scaling method
        minmax = MinMaxScaler()
        standard = StandardScaler()
        robust = RobustScaler()
        normalizer = Normalizer()
        scaling_method = [None, minmax, standard, robust, normalizer]
        scaling_names = ["Original Data", "MinMaxScaler",
                         "StandardScaler", "RobustScaler", "Normalizer"]
        # for showing the number of trials
        i = 0
        for train_index, test_index in kf.split(features, labels):
            # for keeping feature data
            feature_train = []
            feature_test = []
            # for keeping result data
            results_train = []
            results_test = []
            i += 1
            print("---- {} trial -----".format(i))
            # split data
            train_data, train_label, test_data, test_label = features[
                train_index], labels[train_index], features[test_index], labels[test_index]
            # get all data
            for scaling in scaling_method:
                # preprocessing
                if scaling == None:
                    train_scaled = train_data
                    test_scaled = test_data
                    # convert array to list to save list data easily
                    train_scaled = train_scaled.tolist()
                    test_scaled = test_scaled.tolist()
                    feature_train.append(train_scaled)
                    feature_test.append(test_scaled)
                else:
                    # Scaling
                    scaling.fit(train_data)
                    train_scaled = scaling.transform(train_data)
                    test_scaled = scaling.transform(test_data)
                    # convert array to list to save list data easily
                    train_scaled = train_scaled.tolist()
                    test_scaled = test_scaled.tolist()
                    feature_train.append(train_scaled)
                    feature_test.append(test_scaled)
                # training  : LinearSVC
                linsvc.fit(train_scaled, train_label)
                result_train = linsvc.score(train_scaled, train_label)
                result_test = linsvc.score(test_scaled, test_label)
                # saving
                results_train.append(result_train)
                results_test.append(result_test)
            # convert list data to numpy array to use slice
            feature_train = np.array(feature_train)
            feature_test = np.array(feature_test)
            print(type(feature_train))
            # plot 4 times with different axes
            for j in range(len(feature_names)):
                # show result
                for k in range(len(results_train)):
                    print("{0:20} : test score : {test:.3f}, train score : {train:.3f}".format(
                        scaling_names[k], test=results_test[k], train=results_train[k]))

                # prepare for plotting
                fig, axes = plt.subplots(1, 5, figsize=(26, 4))
                # plot features data
                for l in range(5):
                    axes[l].scatter(feature_train[l, :, j % 4], feature_train[l, :, (j+1) % 4],
                                    c=mglearn.cm2(0), label="Training set", s=60)
                    axes[l].scatter(feature_test[l, :, j % 4], feature_test[l, :, (j+1) % 4], marker="^",
                                    c=mglearn.cm2(1), label="Test set", s=60)
                    axes[l].set_title(scaling_names[l])
                axes[0].legend(loc='upper left')
                # make axes labels
                for ax in axes:
                    ax.set_xlabel(feature_names[j % 4])
                    ax.set_ylabel(feature_names[(j+1) % 4])
                plt.show()
                if j < len(feature_names)-1:
                    print("====== Next Features' Plot ======  ")

    def PlotFeatureHistgram(self) -> None:
        """Plot Iris feature data with histogram
        """
        # load iris data
        iris = self.iris
        features, labels, feature_names = iris.data, iris.target, iris.feature_names
        fig, axes = plt.subplots(4, 1, figsize=(20, 32))
        setosa = features[labels == 0]
        versicolor = features[labels == 1]
        virginica = features[labels == 2]

        ax = axes.ravel()

        for i in range(4):
            _, bins = np.histogram(features[:, i], bins=50)
            ax[i].hist(setosa[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
            ax[i].hist(versicolor[:, i], bins=bins,
                       color=mglearn.cm3(1), alpha=.5)
            ax[i].hist(virginica[:, i], bins=bins,
                       color=mglearn.cm3(2), alpha=.5)
            ax[i].set_title(feature_names[i])
            ax[i].set_yticks(())
        ax[0].set_xlabel('Feature magnitude')
        ax[1].set_ylabel('Frequency')
        ax[0].legend(["setosa", "Versicolor", "virginica"])
        fig.tight_layout()

    def PlotPCA(self, n_components: int = 2) -> (list, list, list):
        """PCA(Principal Components Analysis)

        Args:
            n_components (int): number of pricipal components  

        Returns:
            list : standirdized features of Iris
            list : n_components' principal fetures
            list : components of PCA
        """
        # load iris data
        iris = self.iris
        features, labels, feature_names, target_names = iris.data, iris.target, iris.feature_names, iris.target_names

        # Standardization
        scaler = StandardScaler()
        scaler.fit(features)
        features_scaled = scaler.transform(features)
        # convert scaled features into dataframe
        x_scaled = pd.DataFrame(features_scaled, columns=feature_names)

        # PCA
        pca = PCA(n_components)
        pca.fit(features_scaled)
        # Transform features to 2 principal features
        features_pca = pca.transform(features_scaled)

        # convert features data afer pca into dataframe
        df_pca = pd.DataFrame(features_pca)
        # print("Original Shape:{}".format(str(features_scaled.shape)))
        # print("Reduced Shape:{}".format(str(features_pca.shape)))

        # Plot data with 2 principal components axes
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(
            features_pca[:, 0], features_pca[:, 1], labels)
        plt.legend(target_names, loc="best")
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')

        # principal components is saved in pca.components_
        # print("PCA component shape: {}".format(pca.components_.shape))
        # print("PCA componet:\n{}".format(pca.components_))

        # show heatmap for confirming importance of each component
        plt.matshow(pca.components_, cmap="viridis")
        plt.yticks([0, 1], ["First component", "Seconda component"])
        plt.colorbar()
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=60, ha="left")
        plt.xlabel("Feature")
        plt.ylabel("Principal componets")

        return x_scaled, df_pca, pca

    """Evaluation of PCA results
    
    For the first principal components, there are positive correlation except sepal width, 
    so sepal width shows inversed features compared to other features.
    For the second principal componet, sepal width has high influence, second is sepal length, 
    and others han little importance. I guess this is because of uniqueness characteristic of 
    sepal width in the first principal component.

    """

    def PlotNMF(self, n_components: int = 2) -> (list, list, list):
        """NMF(Non-Negative Matrix Factorization)

        Args:
            n_components (int): number of pricipal components  

        Returns:
            list : standirdized features of Iris
            list : n_components' principal fetures
            list : components of NMF
        """
        # load iris data
        iris = self.iris
        features, labels, feature_names, target_names = iris.data, iris.target, iris.feature_names, iris.target_names
        # Standardization
        scaler = StandardScaler()
        scaler.fit(features)
        features_scaled = scaler.transform(features)
        # convert scaled features into dataframe : but not used here
        x_scaled = pd.DataFrame(features_scaled, columns=feature_names)

        # NMF : components must be positive or 0
        nmf = NMF(n_components, random_state=0)
        nmf.fit(features)
        features_nmf = nmf.transform(features)
        # dataframe
        df_nmf = pd.DataFrame(features_nmf)

        # Plot data with 2 principal components axes
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(
            features_nmf[:, 0], features_nmf[:, 1], labels)
        plt.legend(target_names, loc="best")
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')

        # principal components is saved in nmf.components_
        # print("PCA component shape: {}".format(pca.components_.shape))
        # print("PCA componet:\n{}".format(pca.components_))

        # show heatmap for confirming importance of each component
        plt.matshow(nmf.components_, cmap="viridis")
        plt.yticks([0, 1], ["First component", "Seconda component"])
        plt.colorbar()
        plt.xticks(range(len(feature_names)),
                   feature_names, rotation=60, ha="left")
        plt.xlabel("Feature")
        plt.ylabel("Principal componets")

        return x_scaled, df_nmf, nmf

        """Evaluation of NMF
        
        We can classify data by not only first component 
        but also second component with NMF.
        For the first component length are highly weighted.
        For the second component features of sepal is more important 
        than ones of petal, and petal width isn't considered 
        in the second component. I'm not sure why these characteristics
        is so interesting.

        """

    def PlotTSNE(self):
        """t-SNE manifold learning algorithm for data exploration
        """
        # load iris data
        iris = self.iris
        features, labels, feature_names, target_names = iris.data, iris.target, iris.feature_names, iris.target_names
        """
        #Standardization : if scaled data, can be classified by vertical boundary, but we do without scaler here
        scaler = StandardScaler()
        scaler.fit(features)
        features_scaled = scaler.transform(features)
        """
        # TSNE
        tsne = TSNE(random_state=42)
        # transform data
        iris_tsne = tsne.fit_transform(features)
        # print(iris_tsne)

        # define colors for appearence
        colors = ["#476A2A", "#7851B8", "#BD3430"]
        # Plot
        plt.figure(figsize=(10, 10))
        plt.xlim(iris_tsne[:, 0].min(), iris_tsne[:, 0].max()+1)
        plt.ylim(iris_tsne[:, 1].min(), iris_tsne[:, 1].max()+1)
        for i in range(len(features)):
            plt.text(iris_tsne[i, 0], iris_tsne[i, 1], str(labels[i]),
                     color=colors[labels[i]],
                     fontdict={"weight": 'bold', 'size': 9})
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")

    def calculate_center(self, features: list, labels: list) -> list:
        """calculate center

        Args:
            features (list): features 
            labels (list): labels

        Returns:
            list: centers of each group
        """
        # for label 0
        x_0 = 0
        y_0 = 0
        count_0 = 0
        # for label 1
        x_1 = 0
        y_1 = 0
        count_1 = 0
        # for label 2
        x_2 = 0
        y_2 = 0
        count_2 = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                x_0 += features[i, 0]
                y_0 += features[i, 1]
                count_0 += 1
            if labels[i] == 1:
                x_1 += features[i, 0]
                y_1 += features[i, 1]
                count_1 += 1
            if labels[i] == 2:
                x_2 += features[i, 0]
                y_2 += features[i, 1]
                count_2 += 1
        center_x_0 = x_0/count_0
        center_y_0 = y_0/count_0
        center_x_1 = x_1/count_1
        center_y_1 = y_1/count_1
        center_x_2 = x_2/count_2
        center_y_2 = y_2/count_2
        center = np.array([[center_x_0, center_y_0], [
                          center_x_1, center_y_1], [center_x_2, center_y_2]])
        return center

    def PlotKMeans(self):
        """Plot KMeans result 
        """
        # load iris data
        iris = self.iris
        # adapt petal width and length as a features
        features, labels, feature_names = iris.data[:,
                                                    2:], iris.target, iris.feature_names
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(features)

        # plot data
        # Predicted_data
        fig, ax = plt.subplots(figsize=(8, 4))
        print("Labels predicted by KMeans method:\n{}".format(kmeans.labels_))
        mglearn.discrete_scatter(
            features[:, 0], features[:, 1], kmeans.labels_, markers="o", ax=ax)
        mglearn.discrete_scatter(
            kmeans.cluster_centers_[
                :, 0], kmeans.cluster_centers_[:, 1], c="w",
            markers="^", markeredgewidth=2, ax=ax)
        plt.show()

        # Real data
        fig, ax = plt.subplots(figsize=(8, 4))
        cluster_centers_label = self.calculate_center(features, labels)
        print("Labels:\n{}".format(labels))
        mglearn.discrete_scatter(
            features[:, 0], features[:, 1], kmeans.labels_, markers="o", ax=ax)
        mglearn.discrete_scatter(
            cluster_centers_label[:, 0], cluster_centers_label[:, 1], c="w",
            markers="^", markeredgewidth=2, ax=ax)
        plt.show()

    def fancy_dendrogram(self, *args, **kwargs) -> None:
        """Show dendrogram with branch length
        """
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    def PlotDendrogram(self, truncate=False) -> None:
        """show dendrogram 
        """
        # load iris data
        iris = self.iris
        # adapt petal width and length as a features
        features, labels, feature_names = iris.data[:,
                                                    2:], iris.target, iris.feature_names

        # generate the linkage matrix
        # Z[i] : [idx1, idx2, dist, sample_count] in i-th iteration
        Z = linkage(features, 'ward')

        # cophenetic correlation coefficient : how much the data relation is preserved 1 is good
        c, coph_dists = cophenet(Z, pdist(features))
        # {"ward":0.8969802186841395,"average":0.8997592839344625,"complete":0.68726798108573}
        print(c)

        """
        #Analyze Z data in confirming the the merge with 3 components is reasonable 
        print(Z[:30])
        print(features[[4,0,17]])
        #cofirm the distance when merged
        print(Z[-5:,2])
        
        idxs = [45, 6, 17]
        plt.figure(figsize=(10, 8))
        plt.scatter(features[:,0],features[:,1])  # plot all points
        plt.scatter(features[idxs,0], features[idxs,1], c='r')  # plot interesting points in red again
        plt.show()
        """

        """ for checking the num of clusters
        #inconsistency
        depth = 5
        incons = inconsistent(Z, depth)
        print("depth = 5 \n",incons[-10:])
        depth = 3 
        incons = inconsistent(Z, depth)
        print("depth = 3 \n",incons[-10:])
        
        #elbow method
        last = Z[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        plt.plot(idxs, last_rev)

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        plt.plot(idxs[:-2] + 1, acceleration_rev)
        plt.show()
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        print("clusters:", k)
        
        #elbow method including acceleration
        """

        # plot
        plt.figure(figsize=(20, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample')
        plt.ylabel('Clusterdistance')

        # calculate full dendrogram
        if not truncate:
            dendrogram(
                Z,
                leaf_rotation=90,  # rotates the x axis labels
                leaf_font_size=8,  # font size for the x axis labels
                color_threshold=0.15*max(Z[:, 2])
            )
            plt.axhline(y=10, linestyle="--", color="k")
            plt.text(1510, 10, "3 clusters", size=20, ha='left', va='center')
            plt.axhline(y=4.3, linestyle="--", color="k")
            plt.text(1510, 4.3, "4 clusters", size=20, ha='left', va='center')
            plt.show()

        else:

            self.fancy_dendrogram(
                Z,
                truncate_mode="lastp",
                p=10,
                leaf_rotation=90,
                leaf_font_size=12,
                show_contracted=True,
                annotate_above=10,  # useful in small plots so annotations don't overlap
                color_threshold=0.15*max(Z[:, 2])
            )
            plt.show()

    def PlotDBSCAN(self, scaling=False, eps: float = 1.0, min_samples: int = 5) -> None:
        """DBSCAN(Density-based Spatial Clustering of applications with Noise)

        Args:
            scaling (bool): whether we standardized features 
            eps (float): distance for deciding neighbors or not
            min_samples(int) : minimum samples for deciding noise or not 
        """
        # load iris data
        iris = self.iris
        # adapt petal width and length as a features
        features, labels, feature_names = iris.data[:,
                                                    2:], iris.target, iris.feature_names

        # Scaling
        if scaling:
            scaler = StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features)
        print("Cluster memberships:\n{}".format(clusters))
        fig, ax = plt.subplots(figsize=(8, 4))
        mglearn.discrete_scatter(
            features[:, 0], features[:, 1], clusters, markers="o", ax=ax)
        plt.xlabel("Feature 2")
        plt.ylabel("Feature 3")
        plt.show()

    """Discussion about KMeans, Dendrogram, DBSCAN

    if the number of classes is clear, KMeans method is good. If not,
    Dendrogram and DBSCAN will be fine. Dendrogram is helpful for deciding
    how many classes are appropriate. DBSCAN can make classes automatically,
    whose size is controlled by eps and min_samples.Moreover we can make 
    complicated classes with DBSCAN.

    if Num of classes clear:
        K-means
    else:
        if visualizing data:
            Dendrogram
        if make complicated classes:
            DBSCAN
    """
