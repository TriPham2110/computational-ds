"""
Tri Pham
12/11/20
Ref:[1] https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset?select=players20.csv
    [2] https://mml-book.github.io/book/mml-book.pdf
    [3] http://www.cs.otago.ac.nz/cosc453/studenttutorials/principalcomponents.pdf
    [4] http://www.cs.ucc.ie/∼dgb/courses/tai/notes/handout12.pdf
    [5] https://sebastianraschka.com/Articles/2015pcain3steps.html
    [6] https://scikit-learn.org/stable/index.html
    [7] https://matplotlib.org
    [8] https://numpy.org
    [9] https://pandas.pydata.org
"""

from itertools import combinations

# Load libraries
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


def model_prediction(X, y, model):
    x_train, x_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=42)
    if 'PolynomialFeatures()' == str(model):
        poly = PolynomialFeatures(degree=2)
        X_ = poly.fit_transform(x_train)
        X_val_ = poly.fit_transform(x_validation)
        model_ = LinearRegression()
        model_.fit(X_, y_train)
        y_prediction = model_.predict(X_val_)
    else:
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_validation)

    return y_validation, y_prediction


def plot_RMSE(name, RMSE_arr, r=21):
    x = [i for i in range(1, r)]
    plt.plot(x, RMSE_arr)
    plt.grid()
    plt.xlabel('No. of attributes')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. No. of attributes ' + name)
    plt.show()


def bar_RMSE(rmse, r2):
    x = np.arange(3)
    fig, axes = plt.subplots(ncols=1, nrows=1)
    plt.ylabel('Value')
    axes.bar(x + 0, rmse, color='b', width=0.25, label='RMSE')
    axes.bar(x + 0.25, r2, color='g', width=0.25, label='R^2')
    axes.set_xticks(x)
    axes.set_xticklabels(['Origin', 'PCA', 'GA'])
    plt.legend()
    plt.show()


def linear_regresion_result(name, x, y):
    print('Linear regression ' + name)
    y_val, y_pred = model_prediction(x, y, LinearRegression())
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print('RMSE metric:', rmse)
    print('r2 score:', r2)
    return rmse, r2


def poly_regression_result(name, x, y):
    print('Polynomial regression (2nd degree) ' + name)
    y_val, y_pred = model_prediction(x, y, PolynomialFeatures(degree=2))
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print('RMSE metric:', rmse)
    print('r2 score:', r2)
    return rmse, r2


def sv_regression_result(name, x, y):
    print('Support vector regression ' + name)
    y_val, y_pred = model_prediction(x, y, SVR(kernel='rbf'))
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print('RMSE metric:', rmse)
    print('r2 score:', r2)
    return rmse, r2


def rf_regression_result(name, x, y):
    print('Random forest regression ' + name)
    y_val, y_pred = model_prediction(x, y, RandomForestRegressor(criterion='mse', n_jobs=-1))
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print('RMSE metric:', rmse)
    print('r2 score:', r2)
    return rmse, r2


def knn(name, x, y):
    RMSE_arr = []
    for i in range(1, x.shape[1] + 1):
        knn = KNeighborsRegressor(n_neighbors=i)

        # For original data, replace Z with x and y_scaled with y
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=42)

        knn.fit(x_train, y_train)
        y_prediction = knn.predict(x_validation)
        RMSE_arr.append(mean_squared_error(y_validation, y_prediction, squared=False))
    plot_RMSE(name, RMSE_arr, r=x.shape[1] + 1)


# Implementation for genetic algorithm
def crossoverUnion(population):
    unionSets = []
    combinationsOfSubsets = list(combinations(population, 2))
    for i in range(len(combinationsOfSubsets)):
        unionSet = set(combinationsOfSubsets[i][0]).union(set(combinationsOfSubsets[i][1]))
        unionData = data.filter(list(unionSet), axis=1)
        unionSets.append(unionData)
    return unionSets


def crossoverIntersection(population):
    intersectionSets = []
    combinationsOfSubsets = list(combinations(population, 2))
    for i in range(len(combinationsOfSubsets)):
        intersectionSet = set(combinationsOfSubsets[i][0]).intersection(set(combinationsOfSubsets[i][1]))
        intersectionData = data.filter(list(intersectionSet), axis=1)
        intersectionSets.append(intersectionData)
    return intersectionSets


def mutation(crossoverSets, cur_data):
    AddRemoveReplace = rn.randint(0, 3)
    randI = rn.randint(0, 25)
    if AddRemoveReplace == 1:  # Add a feature
        randSetOfFeature = crossoverSets[randI].sample(1, axis=1)
        if not set(randSetOfFeature.columns).issubset(set(cur_data.columns.unique())):
            cur_data = pd.concat([cur_data, randSetOfFeature], axis=1, sort=False)
    if AddRemoveReplace == 0:  # Delete a feature
        randSetOfFeature = crossoverSets[randI].sample(1, axis=1)
        if set(randSetOfFeature.columns).issubset(set(cur_data.columns.unique())) and cur_data.shape[1] > 1:
            cur_data.drop([randSetOfFeature.columns[0]], axis=1, inplace=True)
    if AddRemoveReplace == 2:  # Replacing a feature
        randSetOfFeature1 = crossoverSets[randI].sample(1, axis=1)
        randSetOfFeature2 = crossoverSets[randI].sample(1, axis=1)
        if set(randSetOfFeature1.columns).issubset(set(cur_data.columns.unique())) and cur_data.shape[1] > 2:
            cur_data = pd.concat([cur_data, randSetOfFeature2], axis=1, sort=False)
            cur_data.drop([randSetOfFeature1.columns[0]], axis=1, inplace=True)
    return cur_data


def fiveBestTuples(dataset, acc):
    fiveBestAccs = []
    fiveBestTuples = []
    for i in range(0, 5):
        maxAccuracy = 0
        for j in range(len(acc)):
            if acc[j] > maxAccuracy:
                maxAccuracy = acc[j]
        acc.remove(maxAccuracy)
        bestCurTuple = [item for item in dataset if item[1] == maxAccuracy]
        fiveBestTuples.append(bestCurTuple[0])
        fiveBestAccs.append(maxAccuracy)
    return fiveBestTuples, fiveBestAccs


if __name__ == '__main__':
    dataset = pd.read_csv("../596/players_20.csv", delimiter=',')
    '''
    init random value to replace nan; range chosen between 5 and 25 to represent low attributes' value for not-so-strong position
    for example, goalkeepers are having high gk_[] values but not too high for pace or shooting
    '''
    for ele in enumerate(['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling',
                          'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']):
        dataset[ele[1]] = dataset[ele[1]].apply(lambda v: rn.randint(5, 25))

    dataset = dataset.reset_index(drop=True)

    dataset = dataset.drop(
        ['id', 'nationality', 'club', 'preferred_foot', 'weak_foot', 'team_position', 'player_positions',
         'contract_valid_until'],
        axis=1)
    dataset.to_csv('player_20_modified.csv')

    data = dataset.drop(['name'], axis=1)
    print(data.shape)
    data.replace(" ", float("NaN"), inplace=True)
    data = data.dropna()
    data = data.reset_index(drop=True)

    array = data.values
    x = array[:, :-1]
    y = array[:, -1]

    # StandardScaler helps center and divide each data point by the standard deviation
    from sklearn.preprocessing import StandardScaler
    from numpy.linalg import eig

    x_scaled = StandardScaler().fit_transform(x)

    mean_vector = np.mean(x_scaled, axis=0)
    covariance_matrix = (x_scaled - mean_vector).T.dot((x_scaled - mean_vector)) / (x_scaled.shape[0] - 1)

    eigenvalues, eigenvectors = eig(covariance_matrix)

    # Calculate PoVs
    sumvariance = np.cumsum(eigenvalues)
    sumvariance /= sumvariance[-1]
    print("PoV's: ", (sumvariance * 100).round(2))

    eigen_pairs = list(zip(eigenvalues, eigenvectors))

    W = np.hstack((eigen_pairs[0][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[1][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[2][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[3][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[4][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[5][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[6][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[7][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[8][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[9][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[10][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[11][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[12][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[13][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[14][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[15][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[16][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[17][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[18][1].reshape(len(data.columns) - 1, 1),
                   eigen_pairs[19][1].reshape(len(data.columns) - 1, 1),))
    Z = x_scaled.dot(W)

    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))

    '''
    Choose between LinearRegression(), PolynomialFeatures(), SVR(kernel='rbf'), RandomForestRegressor(n_jobs=-1), KNeighborsRegressor()
    '''
    model = LinearRegression()

    # Set up initial population
    df = pd.DataFrame.copy(data)
    set1 = df.iloc[:, :10]
    set2 = df.iloc[:, :20]
    set3 = df.iloc[:, :30]
    set4 = df.iloc[:, :40]
    set5 = df.iloc[:, :-1]
    population = [set1, set2, set3, set4, set5]

    crossoverSets = []
    cur_population = []
    for i in range(1, 16):
        tupleSetsAndAcc = []
        mutatedSets = []
        accuracy = []
        totalSets = []
        if i == 1:
            cur_population = population
            unionSets = crossoverUnion(cur_population)
            intersectionSets = crossoverIntersection(cur_population)
            crossoverSets = unionSets + intersectionSets + population
            for j in range(0, len(crossoverSets)):
                mutatedSet = mutation(crossoverSets, cur_population[rn.randint(0, 5)])
                mutatedSets.append(mutatedSet)
            totalSets = crossoverSets + mutatedSets
            for individual in totalSets:
                y_validation, y_prediction = model_prediction(individual.values, y, model)
                new_accuracy = r2_score(y_validation, y_prediction)
                accuracy.append(new_accuracy)
            tupleList = list(zip(totalSets, accuracy))
            copied_acc = accuracy
            tupleSetsAndAcc, fiveBestAccuracies = fiveBestTuples(tupleList, copied_acc)
            print("Generation #{:>2}/{:>2} : ".format(1, 15))
            print("Features and r^2 for 5 best sets of features: ")
            for item in tupleSetsAndAcc:
                print("\t", sorted(item[0].columns), end=" & r^2: ")
                print(item[1])
            bestCurSets = [item for item in tupleSetsAndAcc if item[1] == max(fiveBestAccuracies)]
            bestAcc_y_validation, bestAcc_prediction = model_prediction(bestCurSets[0][0].values, y, model=model)
            cur_population = [item[0] for item in tupleSetsAndAcc]
        else:
            unionSets = crossoverUnion(cur_population)
            intersectionSets = crossoverIntersection(cur_population)
            crossoverSets = unionSets + intersectionSets + cur_population
            for j in range(0, len(crossoverSets)):
                mutatedSet = mutation(crossoverSets, cur_population[rn.randint(0, 5)])
                mutatedSets.append(mutatedSet)
            totalSets = crossoverSets + mutatedSets
            for individual in totalSets:
                y_validation, y_prediction = model_prediction(individual.values, y, model=model)
                new_accuracy = r2_score(y_validation, y_prediction)
                accuracy.append(new_accuracy)
            tupleList = list(zip(totalSets, accuracy))
            copied_acc = accuracy
            tupleSetsAndAcc, fiveBestAccuracies = fiveBestTuples(tupleList, copied_acc)
            print("Generation #{:>2}/{:>2} : ".format(i, 15))
            print("Features and r^2 for 5 best sets of features: ")
            for item in tupleSetsAndAcc:
                print("\t", sorted(item[0].columns), end=" & r^2: ")
                print(item[1])
            bestCurSets = [item for item in tupleSetsAndAcc if item[1] == max(fiveBestAccuracies)]
            bestAcc_y_validation, bestAcc_prediction = model_prediction(bestCurSets[0][0].values, y, model=model)
            cur_population = [item[0] for item in tupleSetsAndAcc]
            if i == 15:
                print('\nList of features used to obtain the final evaluation metric: ',
                      list(cur_population[0].columns), len(cur_population[0].columns),
                      '\n\n')

    data_ga = data.filter(list(cur_population[0].columns), axis=1)
    x_ga = data_ga.values

    if 'LinearRegression()' == str(model):
        rmse_original, r2_original = linear_regresion_result('with original data', x, y)
        rmse_pca, r2_pca = linear_regresion_result('with data resulted from PCA', Z, y_scaled)
        rmse_ga, r2_ga = linear_regresion_result('with data resulted from Genetic Algorithm', x_ga, y)
        bar_RMSE([rmse_original, rmse_pca, rmse_ga], [r2_original, r2_pca, r2_ga])
    elif 'PolynomialFeatures()' == str(model):
        rmse_original, r2_original = poly_regression_result('with original data', x, y)
        rmse_pca, r2_pca = poly_regression_result('with data resulted from PCA', Z, y_scaled)
        rmse_ga, r2_ga = poly_regression_result('with data resulted from Genetic Algorithm', x_ga, y)
        bar_RMSE([rmse_original, rmse_pca, rmse_ga], [r2_original, r2_pca, r2_ga])
    elif "SVR()" == str(model):
        rmse_original, r2_original = sv_regression_result('with original data', x, y)
        rmse_pca, r2_pca = sv_regression_result('with data resulted from PCA', Z, y_scaled)
        rmse_ga, r2_ga = sv_regression_result('with data resulted from Genetic Algorithm', x_ga, y)
        bar_RMSE([rmse_original, rmse_pca, rmse_ga], [r2_original, r2_pca, r2_ga])
    elif 'RandomForestRegressor(n_jobs=-1)' == str(model):
        rmse_original, r2_original = rf_regression_result('with original data', x, y)
        rmse_pca, r2_pca = rf_regression_result('with data resulted from PCA', Z, y_scaled)
        rmse_ga, r2_ga = rf_regression_result('with data resulted from Genetic Algorithm', x_ga, y)
        bar_RMSE([rmse_original, rmse_pca, rmse_ga], [r2_original, r2_pca, r2_ga])
    elif 'KNeighborsRegressor()' == str(model):
        knn('for PCA', Z, y_scaled)
        knn('for Original', x, y)
        knn('for GA', x_ga, y)
