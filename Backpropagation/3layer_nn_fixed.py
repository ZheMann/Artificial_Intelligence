import numpy as np
import random
import matplotlib.pyplot as plt

class NeuralNetwork:
    # Learning rate
    ALPHA = 0.3

    # Hidden weights: w1, w2, w3, w4 + b1
    # Output weights: w5, w6, w7 w8 + b2
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = b1 = b2 = 0

    # Storing last gradients for debugging purposes
    gradient_w1 = gradient_w2 = gradient_w3 = gradient_w4 = 0
    gradient_w5 = gradient_w6 = gradient_w7 = gradient_w8 = 0

    def __init__(self, hidden_weights, output_weights, bias_weights):
        # Hidden weights
        self.w1 = hidden_weights[0]
        self.w2 = hidden_weights[1]
        self.w3 = hidden_weights[2]
        self.w4 = hidden_weights[3]

        # Output weights
        self.w5 = output_weights[0]
        self.w6 = output_weights[1]
        self.w7 = output_weights[2]
        self.w8 = output_weights[3]

        # Bias weights
        self.b1 = bias_weights[0]
        self.b2 = bias_weights[1]

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_prime(self, x):
        return x * (1 - x)

    def cost(self, yHat, y):
        return 0.5 * (yHat - y)**2

    def cost_prime(self, yHat, y):
        return -(yHat - y)

    def print_weights(self):
        print("w1:{0}\n"
              "w2:{1}\n"
              "w3:{2}\n"
              "w4:{3}\n"
              "w5:{4}\n"
              "w6:{5}\n"
              "w7:{6}\n"
              "w8:{7}\n"
              "b1:{8}\n"
              "b2:{9}\n".format(self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8, self.b1, self.b2))

    def feed_forward(self, input, prediction_only=False):
        x1 = input[0]
        x2 = input[1]

        net_h1 = self.w1*x1 + self.w2*x2 + self.b1 * 1
        out_h1 = self.logistic(net_h1)

        net_h2 = self.w3*x1 + self.w4*x2 + self.b1 * 1
        out_h2 = self.logistic(net_h2)

        net_o1 = self.w5*out_h1 + self.w6*out_h2 + self.b2 * 1
        out_o1 = self.logistic(net_o1)

        net_o2 = self.w7*out_h1 + self.w8*out_h2 + self.b2*1
        out_o2 = self.logistic(net_o2)

        if prediction_only:
            return [out_o1, out_o2]

        return [out_h1, out_h2, out_o1, out_o2]

    def predict(self, input):
        [y1, y2] = self.feed_forward(input, True)
        y1 = 0 if y1 < 0.5 else 1
        y2 = 0 if y2 < 0.5 else 1

        return [y1, y2]

    def calculate_gradients(self, input, output):
        # Feed forward to calculate outputs of neurons
        [out_hidden1, out_hidden2, out_output1, out_output2] = self.feed_forward(input)

        x1 = input[0]
        x2 = input[1]
        y1 = output[0]
        y2 = output[1]

        # (1) Partial derivative: total error w.r.t. error of output1
        # (2) Partial derivative: total error w.r.t. error of output2
        pd_error_total_wrt_out_output1 = self.cost_prime(y1, out_output1)
        pd_error_total_wrt_out_output2 = self.cost_prime(y2, out_output2)

        # (1) Partial derivative: error output1 w.r.t. net of output1
        # (2) Partial derivative: error output2 w.r.t. net of output2
        pd_out_output1_wrt_net_output1 = self.logistic_prime(out_output1)
        pd_out_output2_wrt_net_output2 = self.logistic_prime(out_output2)

        # (1) Partial derivative: net output 1 w.r.t. w5
        # (2) Partial derivative: net output 1 w.r.t. w6
        # (3) Partial derivative: net output 2 w.r.t. w7
        # (4) Partial derivative: net output 2 w.r.t. w8
        # [net output1 = w5*out_h1 + w6*out_h2 + b2*1]
        # [net output2 = w7*out_h1 + w8*out_h2 + b2*1]
        pd_net_output1_wrt_w5 = out_hidden1
        pd_net_output1_wrt_w6 = out_hidden2
        pd_net_output2_wrt_w7 = out_hidden1
        pd_net_output2_wrt_w8 = out_hidden2

        # The actual gradients:
        # (1) Partial derivative: total error w.r.t  w5
        # (2) Partial derivative: total error w.r.t. w6
        # (3) Partial derivative: total error w.r.t. w7
        # (4) Partial derivative: total error w.r.t. w8
        pd_error_total_wrt_w5 = pd_error_total_wrt_out_output1 * pd_out_output1_wrt_net_output1 * pd_net_output1_wrt_w5
        pd_error_total_wrt_w6 = pd_error_total_wrt_out_output1 * pd_out_output1_wrt_net_output1 * pd_net_output1_wrt_w6
        pd_error_total_wrt_w7 = pd_error_total_wrt_out_output2 * pd_out_output2_wrt_net_output2 * pd_net_output2_wrt_w7
        pd_error_total_wrt_w8 = pd_error_total_wrt_out_output2 * pd_out_output2_wrt_net_output2 * pd_net_output2_wrt_w8


        #####################################
        # START UPDATE WEIGHTS HIDDEN LAYER #
        #                                   #
        #                                   #######################################
        # Each hidden layer neuron output contributes to multiple output neurons. #
        # out_hidden1 affects both out_output1 and out_output2. Therefore total   #
        # error w.r.t. out_hidden1 needs to take both effects into consideration. #                          #
        ###########################################################################

        # (1) Partial derivative: net output1 w.r.t. out hidden1
        # (2) Partial derivative: net output2 w.r.t. out hidden2
        # (1) Partial derivative: net output1 w.r.t. out hidden1
        # (2) Partial derivative: net output2 w.r.t. out hidden2
        # [net output1 = w5*out_h1 + w6*out_h2 + b2*1]
        # [net output2 = w7*out_h1 + w8*out_h2 + b2*1]
        pd_net_output1_out_hidden1 = self.w5
        pd_net_output2_out_hidden1 = self.w7
        pd_net_output1_out_hidden2 = self.w6
        pd_net_output2_out_hidden2 = self.w8

        # (1) Partial derivative: error of output1 w.r.t. net output1
        # (2) Partial derivative: error of output2 w.r.t. net output2
        pd_error_output1_wrt_net_output1 = pd_error_total_wrt_out_output1 * pd_out_output1_wrt_net_output1
        pd_error_output2_wrt_net_output2 = pd_error_total_wrt_out_output2 * pd_out_output2_wrt_net_output2

        # (1) Partial derivative: error of output1 w.r.t. out hidden1
        # (2) Partial derivative: error of output2 w.r.t. out hidden1
        # (3) Partial derivative: error of output1 w.r.t. out hidden2
        # (4) Partial derivative: error of output2 w.r.t. out hidden2
        pd_error_output1_wrt_out_hidden1 = pd_error_output1_wrt_net_output1 * pd_net_output1_out_hidden1
        pd_error_output2_wrt_out_hidden1 = pd_error_output2_wrt_net_output2 * pd_net_output2_out_hidden1
        pd_error_output1_wrt_out_hidden2 = pd_error_output1_wrt_net_output1 * pd_net_output1_out_hidden2
        pd_error_output2_wrt_out_hidden2 = pd_error_output2_wrt_net_output2 * pd_net_output2_out_hidden2

        # (1) Partial derivative: total error w.r.t. out hidden1
        # (2) Partial derivative: total error w.r.t. out hidden2
        pd_error_total_wrt_out_hidden1 = pd_error_output1_wrt_out_hidden1 + pd_error_output2_wrt_out_hidden1
        pd_error_total_wrt_out_hidden2 = pd_error_output1_wrt_out_hidden2 + pd_error_output2_wrt_out_hidden2

        # (1) Partial derivative: out hidden1 w.r.t. net hidden1
        # (2) Partial derivative: out hidden2 w.r.t. net hidden2
        pd_out_hidden1_wrt_net_hidden1 = self.logistic_prime(out_hidden1)
        pd_out_hidden2_wrt_net_hidden2 = self.logistic_prime(out_hidden2)

        # (1) Partial derivative: net hidden1 w.r.t. w1
        # (2) Partial derivative: net hidden1 w.r.t. w2
        # (3) Partial derivative: net hidden2 w.r.t. w3
        # (4) Partial derivative: net hidden2 w.r.t. w4
        # [net hidden1 = w1*x1 + w2*x2 + b1*1]
        # [net hidden2 = w3*x1 + w4*x2 + b1*1]
        pd_net_hidden1_wrt_w1 = x1
        pd_net_hidden1_wrt_w2 = x2
        pd_net_hidden2_wrt_w3 = x1
        pd_net_hidden2_wrt_w4 = x2

        # The actual gradients:
        # (1) Partial derivative: total error w.r.t  w1
        # (2) Partial derivative: total error w.r.t. w2
        # (3) Partial derivative: total error w.r.t. w3
        # (4) Partial derivative: total error w.r.t. w4
        pd_error_total_wrt_w1 = pd_error_total_wrt_out_hidden1 * pd_out_hidden1_wrt_net_hidden1 * pd_net_hidden1_wrt_w1
        pd_error_total_wrt_w2 = pd_error_total_wrt_out_hidden1 * pd_out_hidden1_wrt_net_hidden1 * pd_net_hidden1_wrt_w2
        pd_error_total_wrt_w3 = pd_error_total_wrt_out_hidden2 * pd_out_hidden2_wrt_net_hidden2 * pd_net_hidden2_wrt_w3
        pd_error_total_wrt_w4 = pd_error_total_wrt_out_hidden2 * pd_out_hidden2_wrt_net_hidden2 * pd_net_hidden2_wrt_w4

        self.gradient_w1 = pd_error_total_wrt_w1
        self.gradient_w2 = pd_error_total_wrt_w2
        self.gradient_w3 = pd_error_total_wrt_w3
        self.gradient_w4 = pd_error_total_wrt_w4
        self.gradient_w5 = pd_error_total_wrt_w5
        self.gradient_w6 = pd_error_total_wrt_w6
        self.gradient_w7 = pd_error_total_wrt_w7
        self.gradient_w8 = pd_error_total_wrt_w8

        return [pd_error_total_wrt_w1,
                pd_error_total_wrt_w2,
                pd_error_total_wrt_w3,
                pd_error_total_wrt_w4,
                pd_error_total_wrt_w5,
                pd_error_total_wrt_w6,
                pd_error_total_wrt_w7,
                pd_error_total_wrt_w8]

    def update_weights(self):
        # Update weights
        self.w1 -= self.ALPHA * self.gradient_w1
        self.w2 -= self.ALPHA * self.gradient_w2
        self.w3 -= self.ALPHA * self.gradient_w3
        self.w4 -= self.ALPHA * self.gradient_w4
        self.w5 -= self.ALPHA * self.gradient_w5
        self.w6 -= self.ALPHA * self.gradient_w6
        self.w7 -= self.ALPHA * self.gradient_w7
        self.w8 -= self.ALPHA * self.gradient_w8

    def train(self, input, output):
        self.calculate_gradients(input, output)
        self.update_weights()

    def calculate_total_cost(self, input, output):
        [pred_y1, pred_y2] = self.feed_forward(input, prediction_only=True)
        return self.cost(output[0], pred_y1) + self.cost(output[1], pred_y2)


if __name__ == '__main__':
    # input and output both represent binary numbers
    # goal is to add one to the input 
    training_set = [
        [[0, 1], [1, 0]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ]

    HW = [0.15, 0.20, 0.25, 0.30]
    OW = [0.40, 0.45, 0.50, 0.55]
    B = [0.35, 0.6]
    nn = NeuralNetwork(HW, OW, B)

    costs = []
    for i in range(20000):
        training_input, training_output = random.choice(training_set)
        nn.train(training_input, training_output)
        costs.append(nn.calculate_total_cost(training_input, training_output))

    for j in range(10):
        X, Y = random.choice(training_set)
        print("Target: {0} Predicted: {1}".format(Y, nn.predict(X)))

    plt.plot(costs)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/