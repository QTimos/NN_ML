##------------------------------------------------------------#
##   Instead of using the oop way of using Node/Neuron class
##       and Connection class.
##   We just put everything into flat numpy arrays for faster
##       calculation and lookup speed.
##------------------------------------------------------------#
#
#
#from __future__ import annotations
#import os
#import sys
#from multiprocessing import Process, Pipe
#
#
#def neural_network_process(conn) -> None:
#    import random
#    import itertools
#    import numpy as np
#    from typing import List, Optional, Any
#    from tensorflow.keras.datasets import mnist
#    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#    X_train = X_train / 255.0
#    X_test = X_test / 255.0
#    LEARNING_RATE = 0.75
#    class Layer:
#        def __init__(
#                self,
#                num_of_nodes_curr: int,
#                num_of_nodes_prev: Optional[int] = None,
#                ) -> None:
#
#            self.num_of_nodes_curr = num_of_nodes_curr
#            self.node_activationsV = np.zeros(num_of_nodes_curr, dtype=np.float64)
#            self.node_biasesV = (np.random.rand(num_of_nodes_curr) - 0.5) * 0.1
#            self.node_gradientsV = np.zeros(num_of_nodes_curr, dtype=np.float64)
#            if num_of_nodes_prev:
#                self.weights_of_inbound_connectionsM = (np.random.rand(num_of_nodes_prev, num_of_nodes_curr) - 0.5) * 0.1
#
#
#    class Network:
#        def __init__(self, layers: List[Layer], num_of_layers: int) -> None:
#            self.layers = layers
#            self.num_of_layers = num_of_layers
#            self.corrV = np.zeros(10, dtype=np.float64)
#            self.predV = np.zeros(10, dtype=np.float64)
#
#        def global_forward_pass_loop(self) -> None:
#            for i in range(1, self.num_of_layers):
#                self._forward_pass_loop(self.layers[i], self.layers[i-1])
#            self._set_predV()
#
#        def _forward_pass_loop(self, layer: Layer, prev_layer: Layer) -> None:
#            weighted_sum = np.dot(prev_layer.node_activationsV, layer.weights_of_inbound_connectionsM) + layer.node_biasesV
#            layer.node_activationsV = self._sigmoid(weighted_sum)
#
#        def _sigmoid(self, value: Any) -> np.float64:
#            return (1 / (1 + np.exp(-value)))
#
#        def _set_corrV(self, corr: np.int64):
#            for i in range(10):
#                if i == corr:
#                    self.corrV[i] = 1.0
#                else:
#                    self.corrV[i] = 0.0
#
#        def _set_predV(self) -> None:
#            self.predV = self.layers[-1].node_activationsV
#
#        def get_corrV(self) -> List[float]:
#            return [round(n, 2) for n in self.corrV]
#
#        def get_predV(self) -> List[float]:
#            return [round(n, 2) for n in self.predV]
#
#        def get_corr(self) -> int:
#            return int(np.argmax(self.corrV))
#
#        def get_pred(self) -> int:
#            return int(np.argmax(self.predV))
#
#        def get_error_diffs(self) -> List[np.float64]:
#            return np.array([self.predV[i] - self.corrV[i] for i in range(len(self.predV))])
#
#        def backpropagate(self) -> None:
#            last_layer_index = self.num_of_layers - 1
#            for x in range(last_layer_index, 0, -1):
#                l = self.layers[x]
#                if x == last_layer_index:
#                        output_error_derV = (2 / l.num_of_nodes_curr) * self.get_error_diffs()
#                        activation_slopeV = l.node_activationsV * (1 - l.node_activationsV)
#                        l.node_gradientsV = (output_error_derV * activation_slopeV)
#                else:
#                    nl = self.layers[x + 1]
#                    next_layer_total_blameV = np.dot(nl.weights_of_inbound_connectionsM, nl.node_gradientsV)
#                    activation_slopeV = l.node_activationsV * (1 - l.node_activationsV)
#                    l.node_gradientsV = next_layer_total_blameV * activation_slopeV
#            for x in range(1, last_layer_index + 1):
#                l = self.layers[x]
#                pl = self.layers[x - 1]
#                l.node_biasesV -= LEARNING_RATE * l.node_gradientsV
#                weighted_gradient = np.outer(pl.node_activationsV, l.node_gradientsV)
#                l.weights_of_inbound_connectionsM -= LEARNING_RATE * weighted_gradient
#
#        def mean_squared_error(self) -> np.float64:
#            n = 10
#            return (sum([(self.corrV[i] - self.predV[i])**2 for i in range(n)])) / n
#
#        def update_input_layer(self, list_of_values: np.ndarray, corr: Optional[np.uint8]) -> None:
#            for i in range(self.layers[0].num_of_nodes_curr):
#                self.layers[0].node_activationsV[i] = list_of_values[i]
#            if corr:
#                self._set_corrV(corr)
#
#        def learn(self, image: np.ndarray, corr: np.uint8) -> None:
#            self.update_input_layer(image, corr)
#            self.global_forward_pass_loop()
#            self.backpropagate()
#
#        def test(self, image: np.ndarray, corr: Optional[np.uint8]) -> None:
#            self.update_input_layer(image, None)
#            self.global_forward_pass_loop()
#            return self.get_pred()
#
#        def test_and_print(self, image: np.ndarray, corr: Optional[np.uint8]) -> None:
#            self.update_input_layer(image)
#            self.global_forward_pass_loop()
#            self.print_results()
#
#        def print_results(self) -> None:
#            print(f"Correct was: {self.get_corr()}\nPredicted: {self.get_pred()}")
#            print(f"Error was: {self.mean_squared_error()}\n")
#
#
#    def learning_and_testing() -> None:
#        num_of_input_params = len(X_train[0]) * len(X_train[0][0])
#        layers = [
#                Layer(num_of_input_params),
#                Layer(16, num_of_nodes_prev=num_of_input_params),
#                Layer(16, num_of_nodes_prev=16),
#                Layer(10, num_of_nodes_prev=16)
#                ]
#        network = Network(layers, 4)
#        test = 0
#        for i in range(len(X_train)):
#            if i % 600 == 0:
#                network.test(list(itertools.chain.from_iterable(X_test[test])), Y_test[test])
#                test += 1
#            else:
#                network.learn(list(itertools.chain.from_iterable(X_train[i])), Y_train[i])
#
#    def make_network() -> Network:
#        num_of_input_params = len(X_train[0]) * len(X_train[0][0])
#        layers = [
#                Layer(num_of_input_params),
#                Layer(16, num_of_nodes_prev=num_of_input_params),
#                Layer(16, num_of_nodes_prev=16),
#                Layer(10, num_of_nodes_prev=16)
#                ]
#        network = Network(layers, 4)
#        return network
#
#    def background_training(network):
#        for i in range(len(X_train)):
#            network.learn(list(itertools.chain.from_iterable(X_train[i])), Y_train[i])
#        for i in range(len(X_test)):
#            network.learn(list(itertools.chain.from_iterable(X_test[i])), Y_test[i])
#
#    network = make_network()
#    background_training(network)
#    while True:
#        try:
#            canvas = conn.recv()
#            pred = network.test(list(itertools.chain.from_iterable(canvas)))
#            conn.send(pred)
#        except EOFError:
#            break
#
#def drawing_process() -> None:
#    draw_conn, network_conn = Pipe()
#    network_process = Process(target=neural_network_process, args=(network_conn,), daemon=True)
#    network_process.start()
#
#    import pygame
#    import numpy as np
#    pygame.init()
#    width = 28
#    height = 28
#    scaler = 50
#    canvas = np.zeros((28, 28))
#
#
#    window = pygame.display.set_mode([width*scaler, height*scaler])
#    clock = pygame.time.Clock()
#    running = True
#    drawing = False
#    while running:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                running = False
#            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#                drawing = True
#            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#                drawing = False
#                draw_conn.send(canvas)
#                msg = "Scanning picture..."
#            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
#                canvas = np.zeros((28, 28))
#                ai_message = "Canvas Cleared"
#            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
#                sys.exit(0)
#        if draw_conn.poll():
#            digit = draw_conn.recv()
#            msg = f"Ai guessed: {digit}\n\n  To retry press: R"
#            window.fill("black")
#            font = pygame.font.SysFont(None, 48)
#            window.blit(msg, (width*scaler)/2, (height*scaler)/2)
#            waiting = True
#            while waiting:
#                for event in pygame.event.get():
#                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
#                        waiting = False
#        if drawing:
#            mx, my = pygame.mouse.get_pos()
#            if  0 <= mx < width*scaler and 0 <= my < height*scaler:
#                rect = pygame.Rect(mx, my, scaler, scaler)
#                pygame.draw.rect(window, "white", rect)
#                canvas[mx//scaler, my//scaler] = 1.0
#        pygame.display.flip()
#        clock.tick(60)
#
#    pygame.quit()
#
#def main() -> None:
#    drawing_process()
#
#if __name__ == "__main__":
#    main()









from __future__ import annotations
import os
import sys
from multiprocessing import Process, Pipe

# ------------------------------------------------------------
# PROCESS 1: BACKPROPAGATION & NUMPY AI ENGINE
# ------------------------------------------------------------
def neural_network_process(conn) -> None:
    # Force TensorFlow to skip GPU hooks to completely prevent library conflicts
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import random
    import itertools
    import numpy as np
    from typing import List, Optional, Any
    from tensorflow.keras.datasets import mnist

    print("[AI Process] Booting up and downloading MNIST dataset...")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    LEARNING_RATE = 0.75

    class Layer:
        def __init__(self, num_of_nodes_curr: int, num_of_nodes_prev: Optional[int] = None) -> None:
            self.num_of_nodes_curr = num_of_nodes_curr
            self.node_activationsV = np.zeros(num_of_nodes_curr, dtype=np.float64)
            self.node_biasesV = (np.random.rand(num_of_nodes_curr) - 0.5) * 0.1
            self.node_gradientsV = np.zeros(num_of_nodes_curr, dtype=np.float64)
            if num_of_nodes_prev:
                self.weights_of_inbound_connectionsM = (np.random.rand(num_of_nodes_prev, num_of_nodes_curr) - 0.5) * 0.1

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
            weighted_sum = np.dot(prev_layer.node_activationsV, layer.weights_of_inbound_connectionsM) + layer.node_biasesV
            layer.node_activationsV = self._sigmoid(weighted_sum)

        def _sigmoid(self, value: Any) -> np.float64:
            return (1 / (1 + np.exp(-value)))

        def _set_corrV(self, corr: np.int64):
            for i in range(10):
                if i == corr:
                    self.corrV[i] = 1.0
                else:
                    self.corrV[i] = 0.0

        def _set_predV(self) -> None:
            self.predV = self.layers[-1].node_activationsV

        def get_corrV(self) -> List[float]: return [round(n, 2) for n in self.corrV]
        def get_predV(self) -> List[float]: return [round(n, 2) for n in self.predV]
        def get_corr(self) -> int: return int(np.argmax(self.corrV))
        def get_pred(self) -> int: return int(np.argmax(self.predV))

        def get_error_diffs(self) -> List[np.float64]:
            return np.array([self.predV[i] - self.corrV[i] for i in range(len(self.predV))])

        def backpropagate(self) -> None:
            last_layer_index = self.num_of_layers - 1
            for x in range(last_layer_index, 0, -1):
                l = self.layers[x]
                if x == last_layer_index:
                        output_error_derV = (2 / l.num_of_nodes_curr) * self.get_error_diffs()
                        activation_slopeV = l.node_activationsV * (1 - l.node_activationsV)
                        l.node_gradientsV = (output_error_derV * activation_slopeV)
                else:
                    nl = self.layers[x + 1]
                    next_layer_total_blameV = np.dot(nl.weights_of_inbound_connectionsM, nl.node_gradientsV)
                    activation_slopeV = l.node_activationsV * (1 - l.node_activationsV)
                    l.node_gradientsV = next_layer_total_blameV * activation_slopeV
            for x in range(1, last_layer_index + 1):
                l = self.layers[x]
                pl = self.layers[x - 1]
                l.node_biasesV -= LEARNING_RATE * l.node_gradientsV
                weighted_gradient = np.outer(pl.node_activationsV, l.node_gradientsV)
                l.weights_of_inbound_connectionsM -= LEARNING_RATE * weighted_gradient

        def mean_squared_error(self) -> np.float64:
            n = 10
            return (sum([(self.corrV[i] - self.predV[i])**2 for i in range(n)])) / n

        def update_input_layer(self, list_of_values: np.ndarray, corr: Optional[np.uint8] = None) -> None:
            for i in range(self.layers[0].num_of_nodes_curr):
                self.layers[0].node_activationsV[i] = list_of_values[i]
            if corr is not None:
                self._set_corrV(corr)

        def learn(self, image: np.ndarray, corr: np.uint8) -> None:
            self.update_input_layer(image, corr)
            self.global_forward_pass_loop()
            self.backpropagate()

        # FIXED: Added default None assignment to match your update_input_layer logic
        def test(self, image: np.ndarray, corr: Optional[np.uint8] = None) -> int:
            self.update_input_layer(image, corr)
            self.global_forward_pass_loop()
            return self.get_pred()

    def make_network() -> Network:
        num_of_input_params = len(X_train[0]) * len(X_train[0][0])
        layers = [
                Layer(num_of_input_params),
                Layer(16, num_of_nodes_prev=num_of_input_params),
                Layer(16, num_of_nodes_prev=16),
                Layer(10, num_of_nodes_prev=16)
                ]
        network = Network(layers, 4)
        return network

    def background_training(network):
        print("[AI Process] Starting model training loops...")
        # Train on first 5000 images to avoid freezing the system startup too long
        for i in range(min(5000, len(X_train))):
            network.learn(list(itertools.chain.from_iterable(X_train[i])), Y_train[i])
        print("[AI Process] Training finished! Ready for client connections.")

    network = make_network()
    background_training(network)

    # Notify drawing process that training is complete
    conn.send("READY_TOKEN")

    while True:
        try:
            canvas = conn.recv()
            pred = network.test(list(itertools.chain.from_iterable(canvas)))
            conn.send(pred)
        except EOFError:
            break


# ------------------------------------------------------------
# PROCESS 2: PYGAME GRAPHICS & INTERFACE
# ------------------------------------------------------------
def drawing_process() -> None:
    draw_conn, network_conn = Pipe()
    # FIXED: Changed set literal syntax {} to tuple () for process creation args
    network_process = Process(target=neural_network_process, args=(network_conn,), daemon=True)
    network_process.start()

    import pygame
    import numpy as np
    pygame.init()
    width = 28
    height = 28
    scaler = 20 # Lowered scaler slightly to fit normal desktop monitors comfortably
    canvas = np.zeros((28, 28))

    window = pygame.display.set_mode([width*scaler, height*scaler])
    pygame.display.set_caption("AI is training...")
    clock = pygame.time.Clock()

    running = True
    drawing = False
    ready_to_predict = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False
                if ready_to_predict:
                    draw_conn.send(canvas)
                    pygame.display.set_caption("Scanning picture...")

            # FIXED: Finished incomplete Spacebar condition block
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                canvas = np.zeros((28, 28))
                pygame.display.set_caption("Canvas Cleared")

        # Read background messages sent from the AI worker pipeline
        if draw_conn.poll():
            msg = draw_conn.recv()
            if msg == "READY_TOKEN":
                ready_to_predict = True
                pygame.display.set_caption("Draw a number!")
            else:
                pygame.display.set_caption(f"AI Prediction: {msg}")

        # Continuously track mouse layout edits when clicking down
        if drawing:
            mx, my = pygame.mouse.get_pos()
            if 0 <= mx < width * scaler and 0 <= my < height * scaler:
                canvas[my // scaler, mx // scaler] = 1.0

        # Draw the current grid elements to the Pygame surface canvas
        window.fill("black")
        for y in range(height):
            for x in range(width):
                if canvas[y, x] > 0:
                    rect = pygame.Rect(x * scaler, y * scaler, scaler, scaler)
                    pygame.draw.rect(window, "white", rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    drawing_process()

