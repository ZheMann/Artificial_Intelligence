from numpy import *


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        actual_x = points[i, 0]
        actual_y = points[i, 1]
        predicted_y = m * actual_x + b

        # Increase totalError with squared difference between actual value and predicted value
        totalError += power(actual_y - predicted_y, 2)
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # For each individual point calculate gradient for b (partial derivative)
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))

        # For each individual point calculate gradient for m (partial derivative)
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    # Perform Batch Gradient Descent
    # Update b and m by subtracting the total gradient times the learning rate
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def step_stochastic_gradient(b_current, m_current, points, learning_rate):
    # First perform random shuffle on the data points
    random.shuffle(points)

    # We can always select first data point as the set is shuffled.
    x = points[0, 0]
    y = points[0, 1]
    #print("[{0} {1}]".format(x, y))

    # Calculate gradients for b and m
    b_gradient = -1 * (y - ((m_current * x) + b_current))
    m_gradient = -1 * (x * (y - ((m_current * x) + b_current)))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, sgd):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        if sgd == 0:
            b, m = step_gradient(b, m, array(points), learning_rate)
        else:
            b, m = step_stochastic_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.00001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 2000
    print("BGD || Starting with b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, 0)
    print("BGD || After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    # Reset b and m
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    print("SGD || Starting with b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, 1)
    print("SGD || After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
    run()
