"""
Name: Ezgi Nihal Subasi
Date: 08.03.2021
Project: Weighted k-Nearest Neighbor Classification

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors


# reading csv file according to path and adds column names for separating class, x and y values
def reading_data(path):
    data = pd.read_csv(path)  # reading csv file
    first_row = []
    # appending first row in a list
    for i in range(0, 3):
        first_row.append(float(data.columns[i]))

    data.loc[-1] = first_row  # adding a row
    data.index = data.index + 1  # shifting index
    data = data.sort_index()  # sorting indexes

    data.columns = ["class", "x", "y"]  # assigning column names
    return data


# calculating euclidean distance of a single point for every point in training data set
def calculate_distance(x, y, training_set):
    calculated_dict = training_set.copy()  # copying data set to new dict
    distances = []  # declaring a list
    # iterating in dataframe which is training data set
    for index, row in training_set.iterrows():
        # calculating euclidean distance
        distance = np.sqrt(np.square(abs(row['x'] - x)) + np.square(abs(row['y'] - y)))
        # preventing divided by zero error
        if distance == 0:
            distance = pow(10, -5)
        distances.append(1 / distance)  # appending inverse distance to the list
    # creating a new column in dict to storing calculated distances
    calculated_dict['distance'] = distances
    return calculated_dict


# predicting class of the point by evaluating total votes of each class according to highest k amount of classes
def my_knn(k, sorted_dict):
    class_dict = {1: 0,
                  2: 0,
                  3: 0}  # creating a new dict for storing total votes of each class
    # calculating total votes of k amount of highest class
    for item in range(0, k):
        if sorted_dict['class'][item] == 1:                 # for class 1
            class_dict[1] += sorted_dict['distance'][item]
        elif sorted_dict['class'][item] == 2:               # for class 2
            class_dict[2] += sorted_dict['distance'][item]
        elif sorted_dict['class'][item] == 3:               # for class 3
            class_dict[3] += sorted_dict['distance'][item]

    max_value = 0  # highest vote
    selected_class = 0  # predicting class
    for key, value in class_dict.items():
        if value > max_value:
            max_value = value
            selected_class = key

    return selected_class


# calculating accuracy score by comparing classes in test data with row of predicted class
def my_accuracy_score(test_data):
    accuracy = 0  # initializing accuracy
    # iterating in test data
    for index, row in test_data.iterrows():
        # detecting if prediction is true
        if row['class'] == row['predicted_class']:
            accuracy += 1  # increasing accuracy
    # calculating percentage value of accuracy
    percentage_accuracy = (accuracy * 100) / len(test_data)
    return percentage_accuracy, accuracy


# visualizing scatter plot version of data
def scatter_plot(plotted_df):
    fig, ax = plt.subplots()
    # defining colors for each class
    colors = {1: 'darkorange', 2: 'deepskyblue', 3: 'rebeccapurple'}
    # selecting x and y axises with class column
    ax.scatter(plotted_df['x'], plotted_df['y'], c=plotted_df['class'].map(colors))
    plt.xlabel('x')  # entering name for x axis
    plt.ylabel('y')  # entering name for y axis
    plt.title("Scatter Plot of Training Data")  # entering a title for plot
    plt.show()  # showing the plot


# printing the accuracy score and error count k
def print_results(k, accuracy, error_count, final_data):
    total_count = len(final_data)  # defining total number of count
    new_accuracy = "{:.2f}".format(accuracy)  # decreasing number of digits that comes after the point
    print(f"k={k : <8}   {new_accuracy : ^4}   {total_count - error_count : >8}/{total_count}")


if __name__ == "__main__":

    training_path = "./data_training.csv"  # path of data training csv file
    data_training = reading_data(training_path)  # reading csv file with adding columns
    scatter_plot(data_training)  # visualizing data with scatter plot

    test_path = "./data_test.csv"  # path of data test csv file
    data_test = reading_data(test_path)  # reading csv file with adding columns

    # declaring list that contains index numbers for fixing them after sorting
    new_index = []
    for item in range(0, len(data_training)):
        new_index.append(item)

    # # #  KNN with Sklearn Implementation  # # #

    print("KNN with Sklearn")
    print("          Accuracy (%)   Error Count")
    print("------------------------------------")

    # assigning train and test features separately for fitting into knn
    X_train, y_train, X_test, y_test = data_training[['x', 'y']], data_training['class'], data_test[['x', 'y']], \
                                       data_test['class']

    k_list = [1, 3, 5, 7, 9, 11, 13, 15]  # list for k neighbors

    for k in k_list:

        # defining and setting parameters for using inverse distance weighted and euclidean
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
        # fitting the classifier to the data
        clf = knn.fit(X_train, y_train)
        # predicting a class for each point that placed in the test data
        y_pred = clf.predict(X_test)
        count = 0  # initializing number of true prediction counts

        #  calculating total number of true prediction counts
        for i in range(0, len(data_test)):
            if y_pred[i] == data_test['class'][i]:
                count += 1

        score = accuracy_score(y_test, y_pred) * 100  # calculating accuracy score
        print_results(k, score, count, data_test)  # prints accuracy score with error count

    # # #  KNN from Scratch Implementation  # # #

    print("\nKNN from Scratch")
    print("          Accuracy (%)   Error Count")
    print("------------------------------------")

    for k in k_list:

        predicted_class = []  # storing all predicted classes in a list in each iteration
        for index, row in data_test.iterrows():
            # calculating euclidean distances for each point
            distance_dict = calculate_distance(row['x'], row['y'], data_training)
            # sorting the distances for finding highest ones
            sorted_list = distance_dict.sort_values('distance', ascending=False)
            sorted_list.index = new_index  # arranging index numbers
            predicted_class.append(my_knn(k, sorted_list))

        new_data = data_test.copy()
        new_data['predicted_class'] = predicted_class
        # calculating accuracy score and number of corrects
        score, corrects = my_accuracy_score(new_data)
        # printing the results
        print_results(k, score, corrects, new_data)

    y = data_training['class'].to_numpy()  # assigning predicted class
    X = data_training[['x', 'y']].to_numpy()  # assigning features

    # creating color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']

    h = 0.02  # step size in the mesh
    for n_neighbors in [1, 25]:

        # plot the decision boundary, it will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        xx_yy_matrix = np.c_[xx.ravel(), yy.ravel()]

        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X, y)

        Z = clf.predict(xx_yy_matrix)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification with Sklearn KNN (k = %i, weights = '%s')"
                  % (n_neighbors, 'distance'))
        plt.xlabel('x')
        plt.ylabel('y')
        # Plot the decision boundary. For that, we will assign a color to each
        plt.show()

        df = pd.DataFrame(xx_yy_matrix, columns=['x', 'y'])

        predicted_class = np.array([])
        i = 0
        for index, row in df.iterrows():
            print(i)
            distance_dict = calculate_distance(row['x'], row['y'], data_training)
            sorted_list = distance_dict.sort_values('distance', ascending=False)
            sorted_list.index = new_index
            predicted_class = np.append(predicted_class, my_knn(n_neighbors, sorted_list))
            i += 1

        Z = np.array(predicted_class)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification from Scratch KNN (k = %i, weights = '%s')"
                  % (n_neighbors, 'distance'))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
