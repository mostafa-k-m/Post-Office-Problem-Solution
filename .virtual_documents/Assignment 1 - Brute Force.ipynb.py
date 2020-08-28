from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")
plt.rcParams["figure.figsize"] = (14,7)
def scatterer(data, post, hue = "blue"):
    f, axes = plt.subplots(1, 1)
    plt.scatter(data[:,0],data[:,1], c= hue)
    plt.scatter(post[0],post[1],c="red")
    return f

def calc_distance(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 +(pt1[1]-pt2[1])**2)**0.5

def brute_force_average(X):
    average_distances = []
    for i in X:
        distances = []
        for j in X:
            distances.append(calc_distance(i, j))
        sum_distances = 0
        count = 0
        for k in distances:
            sum_distances += k
            count +=1
        average_distance = sum_distances/count
        average_distances.append(average_distance)
    iterable = 0
    min_distance = np.inf
    for i in average_distances:
        if i< min_distance:
            min_distance = i
            index = iterable
        iterable += 1
    post_location = X[index,:]
    figure = scatterer(X, post_location)
    return post_location, figure


X,y = make_blobs(centers=1, n_samples=100, random_state=60)
brute_force_average(X)


def brute_force_maximum(X):
    max_distances = []
    for i in X:
        max_distance = 0
        for j in X:
            distance = calc_distance(i, j)
            if distance > max_distance:
                max_distance = distance
        max_distances.append(max_distance)
    iterable = 0
    min_distance = np.inf
    for i in max_distances:
        if i< min_distance:
            min_distance = i
            index = iterable
        iterable += 1
    post_location = X[index,:]
    figure = scatterer(X, post_location)
    return post_location, figure


brute_force_maximum(X)


def efficient_distance_calculator(data, point):
    step_1 = (data - point)**2
    step_2 = np.sqrt(step_1.sum(axis=1))
    return step_2



def efficient_average(X):
    minimum = np.inf
    for i in X:
        distances = efficient_distance_calculator(X, i)
        average_distance = sum(distances)/len(distances)
        if average_distance<minimum:
            minimum = average_distance
            post_location = i
    return post_location


post_location = efficient_average(X)
figure = scatterer(X, post_location)


def efficient_maximum(X):
    minimum = np.inf
    for i in X:
        distances = efficient_distance_calculator(X, i)
        maximum_distance = np.max(distances)
        if maximum_distance<minimum:
            minimum = maximum_distance
            post_location = i
    return post_location


post_location = efficient_maximum(X)
figure = scatterer(X, post_location)


X = []
for i in range(1,101):
    X_,y = make_blobs(centers=1, n_samples=10*i, random_state=60)
    X.append(X_)


import timeit, functools

times_for_avg = []
for i in X:
    t = timeit.Timer(functools.partial(efficient_average, i))
    elapsed_time = t.timeit(10)/10
    times_for_avg.append(elapsed_time)


times_for_max = []
for i in X:
    t = timeit.Timer(functools.partial(efficient_maximum, i))
    elapsed_time = t.timeit(10)/10
    times_for_max.append(elapsed_time)


x = list(range(10,1001,10))
plt.plot(x,times_for_avg, label="Minimizing Average Distance")
plt.plot(x,times_for_max, label="Minimizing Maximum Distance")
plt.xlabel('Number of points (n)')
plt.ylabel('Time in ms')
plt.title('Order of growth')
plt.legend(loc="upper left")
