#------------------------------------------------------------#
#   Instead of using the oop way of using Node/Neuron class
#       and Connection class.
#   We just put everything into flat numpy arrays for faster
#       calculation and lookup speed.
#------------------------------------------------------------#
from __future__ import annotations
from typing import List, Optional, Any, Tuple
import os
import sys
from multiprocessing import Process, Pipe


def neural_network_process(conn) -> None:
    import random
    import itertools
    import numpy as np
    from typing import List, Optional, Any
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    BATCH_SIZE = 64
    LEARNING_RATE = 0.2
    class Layer:
        def __init__(
                self,
                num_of_nodes_curr: int,
                num_of_nodes_prev: Optional[int] = None,
                ) -> None:
            self.num_of_nodes_curr = num_of_nodes_curr
            self.node_activationsV = np.zeros(num_of_nodes_curr, dtype=np.float64)
            if num_of_nodes_prev:
                std = np.sqrt(2.0 / num_of_nodes_prev)
                self.weights_of_inbound_connectionsM = np.random.randn(num_of_nodes_prev, num_of_nodes_curr) * std
                self.node_biasesV = np.zeros(num_of_nodes_curr, dtype=np.float64)
            self.node_gradientsV = np.zeros(num_of_nodes_curr, dtype=np.float64)


    class Network:
        def __init__(self, layers: List[Layer], num_of_layers: int) -> None:
            self.layers = layers
            self.num_of_layers = num_of_layers
            self.corrV = np.zeros(10, dtype=np.float64)
            self.predV = np.zeros(10, dtype=np.float64)

        def _forward_pass_loop(self, layer: Layer, prev_layer: Layer, is_output=False) -> None:
            weighted_sum = np.dot(prev_layer.node_activationsV, layer.weights_of_inbound_connectionsM) + layer.node_biasesV
            if is_output:
                exp_vals = np.exp(weighted_sum - np.max(weighted_sum))
                layer.node_activationsV = exp_vals / np.sum(exp_vals)
            else:
                layer.node_activationsV = self._relu(weighted_sum)
            layer.node_rawV = weighted_sum

        def global_forward_pass_loop(self) -> None:
            for i in range(1, self.num_of_layers):
                is_output = (i == self.num_of_layers - 1)
                self._forward_pass_loop(self.layers[i], self.layers[i-1], is_output)
            self._set_predV()

        def _relu(self, value: Any) -> Any:
            return np.maximum(0, value)

        def _relu_derivative(self, value: Any) -> Any:
            return (value > 0).astype(np.float64)

        def _softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def _set_corrV(self, corr: np.int64):
            self.corrV.fill(0)
            self.corrV[corr] = 1.0

        def _set_predV(self) -> None:
            self.predV = self.layers[-1].node_activationsV

        def get_corrV(self) -> List[float]:
            return [round(n, 2) for n in self.corrV]

        def get_predV(self) -> List[float]:
            return [round(n, 2) for n in self.predV]

        def get_corr(self) -> int:
            return int(np.argmax(self.corrV))

        def get_pred(self) -> int:
            return int(np.argmax(self.predV))

        def get_error_diffs(self) -> List[np.float64]:
            return self.predV - self.corrV

        def backpropagate(self) -> None:
            last_layer_index = self.num_of_layers - 1
            for x in range(last_layer_index, 0, -1):
                l = self.layers[x]
                if x == last_layer_index:
                    l.node_gradientsV = self.get_error_diffs()
                else:
                    nl = self.layers[x + 1]
                    next_layer_total_blameV = np.dot(nl.weights_of_inbound_connectionsM, nl.node_gradientsV)
                    activation_slopeV = self._relu_derivative(l.node_rawV)
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

        def cross_entropy_loss(self) -> np.float64:
            return -np.sum(self.corrV * np.log(self.predV + 1e-10))

        def update_input_layer(self, list_of_values: np.ndarray, corr: Optional[np.uint8]) -> None:
            for i in range(self.layers[0].num_of_nodes_curr):
                self.layers[0].node_activationsV[i] = list_of_values[i]
            if corr is not None:
                self._set_corrV(corr)

        def learn(self, image: np.ndarray, corr: np.uint8) -> None:
            self.update_input_layer(image, corr)
            self.global_forward_pass_loop()
            self.backpropagate()

        def forward_batch(self, images):
            self.layers[0].node_activationsM = images
            for i in range(1, self.num_of_layers):
                l = self.layers[i]
                pl = self.layers[i - 1]
                l.raw = pl.node_activationsM @ l.weights_of_inbound_connectionsM + l.node_biasesV
                is_output = (i == self.num_of_layers - 1)
                if is_output:
                    exp_vals = np.exp(l.raw - np.max(l.raw, axis=1, keepdims=True))
                    l.node_activationsM = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
                else:
                    l.node_activationsM = self._relu(l.raw)
            return self.layers[-1].node_activationsM

        def backpropagate_batch(self, labels, batch_size):
            corrM = np.zeros((batch_size, 10))
            corrM[np.arange(batch_size), labels] = 1.0
            last = self.num_of_layers - 1
            for x in range(last, 0, -1):
                l = self.layers[x]
                if x == last:
                    l.node_gradientsM = l.node_activationsM - corrM
                else:
                    nl = self.layers[x + 1]
                    blameM = nl.node_gradientsM @ nl.weights_of_inbound_connectionsM.T
                    l.node_gradientsM = blameM * self._relu_derivative(l.raw)
            for x in range(1, last + 1):
                l = self.layers[x]
                pl = self.layers[x - 1]
                l.weights_of_inbound_connectionsM -= LEARNING_RATE * (pl.node_activationsM.T @ l.node_gradientsM) / batch_size
                l.node_biasesV -= LEARNING_RATE * l.node_gradientsM.mean(axis=0)

        def test(self, image: np.ndarray, corr: Optional[np.uint8] = None) -> int:
            self.update_input_layer(image, None)
            self.global_forward_pass_loop()
            return self.get_pred()

    def make_network() -> Network:
        num_of_input_params = 28 * 28
        layers = [
                Layer(num_of_input_params),
                Layer(256, num_of_nodes_prev=num_of_input_params),
                Layer(128, num_of_nodes_prev=256),
                Layer(64, num_of_nodes_prev=128),
                Layer(10, num_of_nodes_prev=64)
                ]
        network = Network(layers, 5)
        return network

    def background_training(network):
        flat_train = X_train.reshape(len(X_train), -1)
        print("Starting training...")
        for epoch in range(10):
            idx = np.random.permutation(len(flat_train))
            total_loss = 0
            num_batches = 0
            for start in range(0, len(flat_train), BATCH_SIZE):
                batch_idx = idx[start:start + BATCH_SIZE]
                images = flat_train[batch_idx]
                labels = Y_train[batch_idx]
                network.forward_batch(images)
                network.backpropagate_batch(labels, len(batch_idx))
                num_batches += 1
            correct = 0
            for idx in range(200):
                pred = network.test(X_test[idx].flatten())
                if pred == Y_test[idx]:
                    correct += 1
            print(f"Epoch {epoch+1}/10: Test accuracy: {correct/200:.2%}")
            if correct/200 > 0.98:
                print("Good enough! Ready to predict.")
                break

    network = make_network()
    background_training(network)
    conn.send("READY_TOKEN")
    while True:
        try:
            canvas = conn.recv()
            pred = network.test(canvas.flatten())
            conn.send(pred)
        except EOFError:
            break


def get_colors(start_color: Tuple[int], end_color: Tuple[int]) -> List[Tuple[int]]:
    steps = 100
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1)
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        gradient.append((r, g, b))
    return gradient


def drawing_process() -> None:
    draw_conn, network_conn = Pipe()
    network_process = Process(target=neural_network_process, args=(network_conn,), daemon=True)
    network_process.start()

    import pygame
    import numpy as np
    pygame.init()
    width = 28
    height = 28
    scaler = 30

    window = pygame.display.set_mode([width*scaler, height*scaler])
    clock = pygame.time.Clock()
    running = True
    drawing = False
    ready_to_predict = False
    font = pygame.font.SysFont(None, int(scaler*2.9))
    digit_colors = get_colors((231, 130, 132), (229, 200, 144))
    msg_colors = get_colors((229, 200, 144), (166, 209, 137))
    can_draw = False
    forward = True
    prev_pos = None

    i = 0
    j = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if ready_to_predict:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    drawing = True
                    prev_pos = None
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                    prev_pos = None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    window.fill("black")
                    can_draw = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    pix_arr = pygame.surfarray.array3d(window)
                    gray = np.dot(pix_arr[..., :3], [0.299, 0.587, 0.114]).T
                    small_arr = gray.reshape(28, scaler, 28, scaler).mean(axis=(1, 3))
                    small_arr = small_arr / 255.0
                    cy, cx = np.average(np.indices(small_arr.shape), weights=small_arr + 1e-6, axis=(1, 2))
                    shift_y = int(round(14 - cy))
                    shift_x = int(round(14 - cx))
                    small_arr = np.roll(small_arr, shift_y, axis=0)
                    small_arr = np.roll(small_arr, shift_x, axis=1)
                    draw_conn.send(small_arr)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if can_draw:
                        window.fill("black")
        if draw_conn.poll():
            msg = draw_conn.recv()
            if msg == "READY_TOKEN":
                ready_to_predict = True
                can_draw = True
                window.fill("black")
            else:
                window.fill("black")
                text = f"AI guessed: {msg}"
                text2 = "Press R to retry"
                color = digit_colors[j % len(digit_colors)]
                text_sur = font.render(text, False, color)
                text_sur2 = font.render(text2, False, color)
                text_rect = text_sur.get_rect(center=((width*scaler)//2, (height*scaler)//2 - scaler//1))
                text_rect2 = text_sur2.get_rect(center=((width*scaler)//2, (height*scaler)//2 + scaler//1))
                window.blit(text_sur, text_rect)
                window.blit(text_sur2, text_rect2)
                can_draw = False
        if ready_to_predict and can_draw:
            if drawing:
                mx, my = pygame.mouse.get_pos()
                if 0 <= mx < width*scaler and 0 <= my < height*scaler:
                    if prev_pos:
                        pygame.draw.line(window, (133, 193, 220), prev_pos, (mx, my), int(scaler * 1.5))
                    pygame.draw.circle(window, (133, 193, 220), (mx, my), int(scaler * 0.8))
                    prev_pos = (mx, my)
        if not ready_to_predict:
            can_draw = False
            text = "AI is training..."
            color = msg_colors[j]
            text_sur = font.render(text, False, color)
            text_rect = text_sur.get_rect(center=((width*scaler)//2, (height*scaler)//2))
            window.blit(text_sur, text_rect)
        pygame.display.flip()
        clock.tick(60)
        i += 1
        if i % 2 == 0:
            if j >= len(msg_colors) - 1:
                forward = False
            if j <= 0:
                forward = True
            if forward:
                j += 1
            else:
                j -= 1
    pygame.quit()


def main() -> None:
    drawing_process()


if __name__ == "__main__":
    main()
