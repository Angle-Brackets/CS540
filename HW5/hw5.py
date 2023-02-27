import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def q3(data):
    X = np.empty((len(data),2)) #Years
    Y = np.empty((len(data))) #Days

    row = 0
    for year, days in data.items():
        x_i = np.array([1, year])
        X[row] = x_i
        Y[row] = days
        row += 1
    
    #Q3a display
    np.set_printoptions(suppress=True)
    print("Q3a: ")
    print(X)

    #Q3b display
    print("Q3b: ")
    print(Y)

    #Matrix product step for Z, Q3c
    Z = np.dot(np.transpose(X), X)
    print("Q3c: ")
    print(Z)

    #Inverse of our matrix product, Q3d
    np.set_printoptions(suppress=False)
    I = np.linalg.inv(Z)
    print("Q3d: ")
    print(I)

    #Psuedo Inverse of X, Q3e
    PI = np.dot(I, np.transpose(X))
    print("Q3e: ")
    print(PI)

    #Beta Hat calculation, Q3f
    hat_beta = np.dot(PI, Y)
    print("Q3f: ")
    print(hat_beta)
    
    return hat_beta

def q4(hat_beta, x_test):
    B0 = hat_beta[0]
    B1 = hat_beta[1]
    
    y_test = B0 + B1*x_test
    
    print(f"Q4: {y_test}")

def q5(hat_beta):
    B1 = hat_beta[1]
    
    print("Q5a: ", end="")
    if B1 < 0:
        print("<")
        print("Q5b: Since the sign is negative for beta 1, this implies that the slope of the line is decreasing over time with our model, implying that the amount of ice is predicted to decrease over time on Lake Mendota")
    elif B1 > 0:
        print(">")
        print("Q5b: Since the sign is positive for beta 1, this implies that the slope of the line is increasing over time with our model, implying that the amount of ice is predicted to increase over time on Lake Mendota")
    else:
        print("=")
        print("Q5b: Since the sign is none for beta 1, this implies that the slope of the line is flat over time with our model, implying that the amount of ice is predicted to stay constant over time on Lake Mendota")

def q6(hat_beta):
    # Just do some basic algebra to get this!
    B0 = hat_beta[0]
    B1 = hat_beta[1]
    x_star = -B0/B1 #The year when lake mendota won't freeze anymore
    
    print(f"Q6a: {x_star}")
    print("Q6b: I would say that this date does not make sense based on prior trends as it is FAR too far in the future to be realistic. It is clear that based on data trends that Lake Mendota freezes for less and less time each year, so this accelerating rate is not adequately captured in this linear model, implying that extrapolating the data this far is inaccurate and the year generated does not make sense given how the data has trended over time.")


def main():
    file = sys.argv[1]
    dataset = dict()

    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset[int(row["year"])] = int(row["days"])

    plt.plot(list(dataset.keys()), list(dataset.values()))
    plt.xlabel("Number of frozen days")
    plt.ylabel("Year")
    plt.savefig("plot.jpg")

    
    hat_beta = q3(dataset)
    q4(hat_beta, 2021)
    q5(hat_beta)
    q6(hat_beta)

    



if __name__ == "__main__":
    main()
