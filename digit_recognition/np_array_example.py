#------------------------------------------------------------#
#   Instead of using the oop way of using Node/Neuron class
#       and Connection class.
#   We just put everything into flat numpy arrays for faster
#       calculation and lookup speed.
#------------------------------------------------------------#


from __future__ import annotations
import random
import itertools
import numpy as np
from typing import List, Optional, Any
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# normalizing the pixels from between 0, 255 to 0.0, 1.0
X_train = X_train / 255.0
X_test = X_test / 255.0
LEARNING_RATE = 5.0

class Layer:
    def __init__(
            self,
            num_of_nodes_curr: int,
            num_of_nodes_prev: Optional[int] = None,
            num_of_nodes_next: Optional[int] = None
            ) -> None:

        self.num_of_nodes_curr = num_of_nodes_curr
        self.node_activations = np.zeros(num_of_nodes_curr, dtype=np.float64)
        self.node_biases = (np.random.rand(num_of_nodes_curr) - 0.5) * 0.1
        self.node_blames = np.zeros(num_of_nodes_curr, dtype=np.float64)
        if num_of_nodes_prev:
            self.weights_matrix_of_inbound_connections = (np.random.rand(num_of_nodes_prev, num_of_nodes_curr) - 0.5) * 0.1
        if num_of_nodes_next:
            self.weights_matrix_of_outbound_connections = (np.random.rand(num_of_nodes_curr, num_of_nodes_next) - 0.5) * 0.1


class Network:
    def __init__(self, layers: List[Layer], num_of_layers: int) -> None:
        self.layers = layers
        self.num_of_layers = num_of_layers
        self.corrV = np.zeros(10, dtype=np.float64)
        self.predV = np.zeros(10, dtype=np.float64)

    def global_forward_pass_loop(self) -> None:
        for i in range(1, self.num_of_layers):
            self._forward_pass_loop(self.layers[i], self.layers[i-1])
        self._set_predV()

    def _forward_pass_loop(self, layer: Layer, prev_layer: Layer) -> None:
        weighted_sum = np.dot(prev_layer.node_activations, layer.weights_matrix_of_inbound_connections) + layer.node_biases
        layer.node_activations = self._sigmoid(weighted_sum)

    def _sigmoid(self, value: Any) -> np.float64:
        return (1 / (1 + np.exp(-value)))

    def _set_corrV(self, corr: np.int64):
        for i in range(10):
            if i == corr:
                self.corrV[i] = 1.0
            else:
                self.corrV[i] = 0.0

    def _set_predV(self) -> None:
        self.predV = self.layers[-1].node_activations

    def get_corrV(self) -> List[float]:
        return [round(n, 2) for n in self.corrV]

    def get_predV(self) -> List[float]:
        return [round(n, 2) for n in self.predV]

    def get_corr(self) -> int:
        return int(np.argmax(self.corrV))

    def get_pred(self) -> int:
        return int(np.argmax(self.predV))

    def get_error_derivatives(self) -> List[np.float64]:
        return np.array([self.predV[i] - self.corrV[i] for i in range(len(self.predV))])

    def backpropagate(self) -> None:
        output_error_der = self.get_error_derivatives()
        last_layer_index = self.num_of_layers - 1
        for x in range(last_layer_index, 0, -1):
            l = self.layers[x]
            if x == last_layer_index:
                    l.node_blames = (
                            ((2 / l.num_of_nodes_curr) * output_error_der)
                            * l.node_activations * (1 - l.node_activations)
                            )
            else:
                nl = self.layers[x + 1]
                total_blame = np.dot(l.weights_matrix_of_outbound_connections, nl.node_blames)
                node_slope = l.node_activations * (1 - l.node_activations)
                l.node_blames = total_blame * node_slope
        for x in range(1, last_layer_index + 1):
            l = self.layers[x]
            pl = self.layers[x - 1]
            l.node_biases -= LEARNING_RATE * l.node_blames
            weighted_gradient = np.outer(pl.node_activations, l.node_blames)
            l.weights_matrix_of_inbound_connections -= LEARNING_RATE * weighted_gradient

    def mean_squared_error(self) -> np.float64:
        n = 10
        return (sum([(self.corrV[i] - self.predV[i])**2 for i in range(n)])) / n

    def update_input_layer(self, list_of_values: np.ndarray, corr: np.uint8) -> None:
        for i in range(self.layers[0].num_of_nodes_curr):
            self.layers[0].node_activations[i] = list_of_values[i]
        self._set_corrV(corr)

    def learn(self, image: np.ndarray, corr: np.uint8) -> None:
        self.update_input_layer(image, corr)
        self.global_forward_pass_loop()
        self.backpropagate()

    def test(self, image: np.ndarray, corr: np.uint8) -> None:
        self.update_input_layer(image, corr)
        self.global_forward_pass_loop()
        self.print_results()

    def print_results(self) -> None:
        print(f"Correct was: {self.get_corr()}\nPredicted: {self.get_pred()}")
        print(f"Error was: {self.mean_squared_error()}\n")




def main() -> None:
    num_of_input_params = len(X_train[0]) * len(X_train[0][0])
    layers = [
            Layer(num_of_input_params, num_of_nodes_next=16),
            Layer(16, num_of_nodes_prev=num_of_input_params, num_of_nodes_next= 16),
            Layer(16, num_of_nodes_prev=16, num_of_nodes_next=10),
            Layer(10, num_of_nodes_prev=16)
            ]
    network = Network(layers, 4)
    test = 0
    for i in range(len(X_train)):
        if i % 600 == 0:
            network.test(list(itertools.chain.from_iterable(X_test[test])), Y_test[test])
            test += 1
        else:
            network.learn(list(itertools.chain.from_iterable(X_train[i])), Y_train[i])





if __name__ == "__main__":
    main()
