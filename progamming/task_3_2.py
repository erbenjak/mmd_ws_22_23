import numpy as np
import sys

def create_function(degree,alphas,x):
    result = 0
    l = len(alphas)
    while (degree > 0):
        result += alphas[l - degree]* x ** (degree-1)
        degree -= 1
        return result


def gradient_descent(degree_of_function, iterations, lr, data):
    print("caclulating gradient for function with degree:"+str(degree_of_function))
    alphas = np.ones(int(degree_of_function))

    # we will use an automatic creation for the derivation
    for iter in range(0, iterations):
        function = lambda x : create_function(degree_of_function,alphas.tolist(),x)

        #calculate the initial MSE:
        x_values = points[0]
        y_values = points[1]

        y_calc_values = np.apply_along_axis(function,0,x_values)
        errors = (y_values-y_calc_values)
        squared_errors = np.square(errors)
        initial_mse = 1/points.shape[1] * np.sum(squared_errors)

        print("The MSE at iteration " +str(iter)+" is:" + str(initial_mse))

        gradients=[]

        # now all the gradients need to be calculated
        current_deg_alpha = degree_of_function
        for alpha in alphas:
            custom_func = lambda x: current_deg_alpha*x^(current_deg_alpha-1)
            factors = np.apply_along_axis(custom_func,0,x_values)
            current_grad= 1/points.shape[1] * np.sum(-2*factors*errors)
            gradients.append(current_grad*lr)
            current_deg_alpha -= 1

        alphas = alphas-np.array(gradients)

    return alphas


if __name__ == '__main__':
    commandline_degree = 0
    for i, arg in enumerate(sys.argv):
        if i == 1:
            commandline_degree = arg

    print("Function we will try to find: 3x³-2x-17");
    points = []
    values =[]
    for i in range(0,10):
        x=i-5
        real_val = 3*x^3-2*x-17
        points.append(x)
        values.append(real_val)
    points = np.array([points,values])
    alphas=gradient_descent(int(commandline_degree), iterations = 100, lr=0.0001, data=points)
    print("alpha values estimated:"+str(alphas))