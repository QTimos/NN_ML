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
LEARNING_RATE = 10.0


class Node:
    def __init__(self, index: int, activation: Optional[np.float64] = None, bias: Optional[np.float64] = None) -> None:
        self.index = index
        self.outbound_connection_list = []
        self.inbound_connection_list = []
        if activation:
            self.activation = activation
        else:
            self.activation = np.float64(round(random.random(), 2))
        if bias:
            self.bias = bias
        else:
            self.bias = np.float64(round(random.random(), 2))
        self.blame = 0


class Connection:
    def __init__(self, n1: Node, n2: Node) -> None:
        self.n1 = n1
        self.n2 = n2
        self.weight = np.float64((random.random() - 0.5) * 0.1)


class Layer:
    def __init__(self, num_of_nodes: int, list_of_values: Optional[List] = None) -> None:
        self.inbound_connections = []
        self.outbound_connections = []
        self.num_of_nodes = num_of_nodes
        self.nodes = self._init_nodes(self.num_of_nodes, list_of_values)

    def _init_nodes(self, num_of_nodes: int, list_of_values: Optional[List] = None) -> List:
        nodes = []
        if list_of_values:
            for i in range(num_of_nodes):
                nodes.append(Node(i, list_of_values[i]))
        else:
            for i in range(num_of_nodes):
                nodes.append(Node(i))
        return nodes

    def init_connections_with_next(self, other_layer: Layer) -> None:
        for i in range(self.num_of_nodes):
            for j in range(other_layer.num_of_nodes):
                connection = Connection(self.nodes[i], other_layer.nodes[j])
                self.outbound_connections.append(connection)
                self.nodes[i].outbound_connection_list.append(connection)
                other_layer.nodes[j].inbound_connection_list.append(connection)


class Network:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self._connect_layers(self.layers)
        self.corrV = []
        self.predV = []

    def _connect_layers(self, layers: List[Layer]) -> None:
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                layers[i].init_connections_with_next(layers[i + 1])
                layers[i + 1].inbound_connections = layers[i].outbound_connections
        else:
            print("Must have more than one layer to connect them.")

    def global_forward_pass_loop(self) -> None:
        for i in range(1, len(self.layers)):
            self._forward_pass_loop(self.layers[i])
        self._set_predV()

    def _forward_pass_loop(self, layer: Layer) -> None:
        for n in layer.nodes:
            n.activation = self._sigmoid(
                    sum(
                        [c.weight * c.n1.activation for c in n.inbound_connection_list],
                        n.bias
                        ))

    def _sigmoid(self, value: Any) -> np.float64:
        return np.float64(1 / (1 + np.exp(-value)))

    def _set_corrV(self, corr) -> List:
        v = []
        for i in range(10):
            if i == corr:
                v.append(np.float64(1.0))
            else:
                v.append(np.float64(0.0))
        return v

    def update_input_layer(self, list_of_values: np.ndarray, corr: np.uint8) -> None:
        for i, n in enumerate(self.layers[0].nodes):
            n.activation = list_of_values[i]
        self.corr = corr
        self.corrV = self._set_corrV(corr)

    def _set_predV(self) -> None:
        self.predV = []
        for n in self.layers[-1].nodes:
            self.predV.append(n.activation)

    def get_corrV(self) -> List[np.float64]:
        return self.corrV

    def get_predV(self) -> List[np.float64]:
        return self.predV

    def get_corr(self) -> np.int64:
        return np.argmax(self.corrV)

    def get_pred(self) -> np.int64:
        return np.argmax(self.predV)

    def mean_squared_error(self) -> np.float64:
        n = len(self.predV)
        return (sum([(self.corrV[i] - self.predV[i])**2 for i in range(n)])) / n

    def get_error_derivatives(self) -> List[np.float64]:
        return list([self.predV[i] - self.corrV[i] for i in range(len(self.predV))])

    def backpropagate(self) -> None:
        output_error_der = self.get_error_derivatives()
        hidden_layers_deltas = []
        last_layer_index = len(self.layers) - 1
        for x in range(last_layer_index, 0, -1):
            if x == last_layer_index:
                for i, n in enumerate(self.layers[x].nodes):
                    n.blame = ((2 / len(self.layers[x].nodes)) * output_error_der[i]) * n.activation * (1 - n.activation)
            else:
                for i, n in enumerate(self.layers[x].nodes):
                    total_blame = sum([c.weight * c.n2.blame for c in n.outbound_connection_list])
                    node_slope = n.activation * (1 - n.activation)
                    n.blame = total_blame * node_slope
        for x in range(1, last_layer_index + 1):
            for i, n in enumerate(self.layers[x].nodes):
                n.bias -= LEARNING_RATE * n.blame
            for c in self.layers[x].inbound_connections:
                c.weight -= LEARNING_RATE * c.n1.activation * c.n2.blame

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
            Layer(num_of_input_params),
            Layer(16),
            Layer(16),
            Layer(10)
            ]
    network = Network(layers)
    test = 0
    for i in range(len(X_train)):
        if i % 20 == 0:
            network.test(list(itertools.chain.from_iterable(X_test[test])), Y_test[test])
            test += 1
        else:
            network.learn(list(itertools.chain.from_iterable(X_train[i])), Y_train[i])


#    print(f"pred: {network.get_pred()}\ncorr: {network.get_corr()}")




if __name__ == "__main__":
    main()
