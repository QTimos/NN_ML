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


class Node:
    def __init__(self, index: int, activation: Optional[np.float64] = None, bias: Optional[int] = None) -> None:
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

    def increment_bias(self) -> None:
        self.bias += 0.1

    def decrement_bias(self) -> None:
        self.bias -= 0.1


class Connection:
    def __init__(self, n1: Node, n2: Node) -> None:
        self.n1 = n1
        self.n2 = n2
        self.weight = np.float64(round(random.random(), 2))

    def increment_weight(self) -> None:
        self.weight += 0.1

    def decrement_weight(self) -> None:
        self.weight -= 0.1


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
                self.nodes[j].inbound_connection_list.append(connection)


class Network:
    def __init__(self, layers: List[Layer], corr: int) -> None:
        self.layers = layers
        self._connect_layers(self.layers)
        self.corrV = self._set_corrV(corr)
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

    def update_input_layer(self, corr: int, list_of_values: Optional[List] = None) -> None:
        for i, n in enumerate(self.layers[0].nodes):
            n.activation = list_of_values[i]
        self.corr = corr
        self.corrV = self._set_corrV(corr)

    def _set_predV(self) -> None:
        self.predV = []
        for n in self.layers[-1].nodes:
            self.predV.append(n.activation)

    def get_corrV(self) -> None:
        return self.corrV

    def get_predV(self) -> None:
        return self.predV

    def get_corr(self) -> None:
        return np.argmax(self.corrV)

    def get_pred(self) -> None:
        return np.argmax(self.predV)



def mean_squared_error(n: int, corrV: List[int], predV: List[int]) -> np.float64:
    return (sum([(corrV[i] - predV[i])**2 for i in range(n)])) / n



def main() -> None:
    num_of_input_params = len(X_train[0]) * len(X_train[0][0])
    list_of_values = list(itertools.chain.from_iterable(X_train[0]))
    layers = [
            Layer(num_of_input_params, list_of_values),
            Layer(16),
            Layer(16),
            Layer(10)
            ]
    network = Network(layers, Y_train[0])
    network.global_forward_pass_loop()


    print(f"pred: {network.get_pred}\ncorr: {network.get_corr}")




if __name__ == "__main__":
    main()
