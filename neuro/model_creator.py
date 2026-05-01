import numpy as np
from .activations import get_activation_function
from .losses import get_loss_function



class MLP:

    def __init__(self, layers, activations, loss='mse', learning_rate=0.01, random_state=None):

        self.layers = layers
        self.activations = activations
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            current_activation = activations[min(i, len(activations) - 1)]

            if current_activation in ['relu', 'leaky_relu']:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            else:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.activation_functions = []
        self.activation_derivatives = []

        for act_name in activations:
            act_func, act_deriv = get_activation_function(act_name)
            self.activation_functions.append(act_func)
            self.activation_derivatives.append(act_deriv)

        self.loss_function, self.loss_derivative = get_loss_function(loss)

        self.history = {'loss': [], 'accuracy': []}

    def forward(self, X):

        activations = [X]
        current_input = X

        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.activation_functions[i](z)
            activations.append(a)
            current_input = a

        return activations

    def backward(self, X, y, activations):
        m = X.shape[0]

        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        z_values = []
        for i in range(len(self.weights)):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)

        if self.loss_name == 'categorical_crossentropy' and self.activations[-1] == 'softmax':
            dz = activations[-1] - y
        else:
            da = self.loss_derivative(y, activations[-1])
            dz = da * self.activation_derivatives[-1](z_values[-1])

        dW[-1] = np.dot(activations[-2].T, dz)
        db[-1] = np.sum(dz, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[i + 1].T)
            dz = da * self.activation_derivatives[i](z_values[i])

            dW[i] = np.dot(activations[i].T, dz)
            db[i] = np.sum(dz, axis=0, keepdims=True)

        return dW, db

    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y, epochs=1000, verbose=True, validation_data=None):

        for epoch in range(epochs):

            activations = self.forward(X)

            loss = self.loss_function(y, activations[-1])
            self.history['loss'].append(loss)

            if self.loss_name in ['binary_crossentropy', 'categorical_crossentropy']:
                if len(y.shape) == 1 or y.shape[1] == 1:
                    predictions = (activations[-1] > 0.5).astype(int)
                    accuracy = np.mean(predictions.flatten() == y.flatten())
                else:
                    predictions = np.argmax(activations[-1], axis=1)
                    true_labels = np.argmax(y, axis=1)
                    accuracy = np.mean(predictions == true_labels)
                self.history['accuracy'].append(accuracy)

            dW, db = self.backward(X, y, activations)

            max_grad_norm = 5.0
            for i in range(len(dW)):
                grad_norm = np.linalg.norm(dW[i])
                if grad_norm > max_grad_norm:
                    dW[i] = dW[i] * max_grad_norm / grad_norm

                grad_norm_b = np.linalg.norm(db[i])
                if grad_norm_b > max_grad_norm:
                    db[i] = db[i] * max_grad_norm / grad_norm_b

            self.update_parameters(dW, db)

            if epoch == 0 and verbose:
                total_grad_norm = sum(np.linalg.norm(dw) for dw in dW)
                print(f"Initial gradient norm: {total_grad_norm:.6f}")

            if verbose and (epoch + 1) % 100 == 0:
                if self.loss_name in ['binary_crossentropy', 'categorical_crossentropy']:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

    def predict_classes(self, X):
        predictions = self.predict(X)
        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)

    def check_gradients(self, X, y, epsilon=1e-7):

        activations = self.forward(X)
        dW_analytical, db_analytical = self.backward(X, y, activations)

        for layer_idx in range(len(self.weights)):
            W = self.weights[layer_idx]
            dW_numerical = np.zeros_like(W)

            indices = [(i, j) for i in range(min(3, W.shape[0])) for j in range(min(3, W.shape[1]))]

            for i, j in indices:
                self.weights[layer_idx][i, j] += epsilon
                activations_plus = self.forward(X)
                loss_plus = self.loss_function(y, activations_plus[-1])

                self.weights[layer_idx][i, j] -= 2 * epsilon
                activations_minus = self.forward(X)
                loss_minus = self.loss_function(y, activations_minus[-1])

                self.weights[layer_idx][i, j] += epsilon

                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                diff = abs(dW_analytical[layer_idx][i, j] - dW_numerical[i, j])
                relative_diff = diff / (abs(dW_analytical[layer_idx][i, j]) + abs(dW_numerical[i, j]) + 1e-8)

                if relative_diff > 1e-5:
                    print(f"Gradient mismatch at layer {layer_idx}, position ({i},{j})")
                    print(f"Analytical: {dW_analytical[layer_idx][i, j]:.8f}")
                    print(f"Numerical: {dW_numerical[i, j]:.8f}")
                    print(f"Relative difference: {relative_diff:.8f}")
                    return False

        print("Gradient check passed!")
        return True


class MLPClassifier(MLP):

    def __init__(self, hidden_layers=[100], activation='relu', output_activation='sigmoid',
                 learning_rate=0.01, random_state=None):

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation

        self.input_size = None
        self.output_size = None

        super().__init__([], [], loss='binary_crossentropy',
                        learning_rate=learning_rate, random_state=random_state)

    def fit(self, X, y, epochs=1000, verbose=True, validation_data=None):

        self.input_size = X.shape[1]

        if len(y.shape) == 1:
            self.output_size = 1
            y = y.reshape(-1, 1)
            loss = 'binary_crossentropy'
        else:
            self.output_size = y.shape[1]
            loss = 'categorical_crossentropy' if y.shape[1] > 1 else 'binary_crossentropy'

        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        activations = [self.activation] * len(self.hidden_layers) + [self.output_activation]

        self.layers = layers
        self.activations = activations
        self.loss_name = loss

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            current_activation = activations[min(i, len(activations) - 1)]
            if current_activation in ['relu', 'leaky_relu']:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            else:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.activation_functions = []
        self.activation_derivatives = []

        for act_name in activations:
            act_func, act_deriv = get_activation_function(act_name)
            self.activation_functions.append(act_func)
            self.activation_derivatives.append(act_deriv)

        self.loss_function, self.loss_derivative = get_loss_function(loss)
        self.history = {'loss': [], 'accuracy': []}

        super().fit(X, y, epochs, verbose, validation_data)


class MLPRegressor(MLP):

    def __init__(self, hidden_layers=[100], activation='relu', learning_rate=0.01, random_state=None):
        self.hidden_layers = hidden_layers
        self.activation = activation

        self.input_size = None
        self.output_size = None

        super().__init__([], [], loss='mse', learning_rate=learning_rate, random_state=random_state)

    def fit(self, X, y, epochs=1000, verbose=True, validation_data=None):

        self.input_size = X.shape[1]
        self.output_size = 1 if len(y.shape) == 1 else y.shape[1]

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        activations = [self.activation] * len(self.hidden_layers) + ['linear']

        self.layers = layers
        self.activations = activations

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            current_activation = activations[min(i, len(activations) - 1)]
            if current_activation in ['relu', 'leaky_relu']:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            else:
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.activation_functions = []
        self.activation_derivatives = []

        for act_name in activations:
            act_func, act_deriv = get_activation_function(act_name)
            self.activation_functions.append(act_func)
            self.activation_derivatives.append(act_deriv)

        self.loss_function, self.loss_derivative = get_loss_function('mse')
        self.history = {'loss': [], 'accuracy': []}

        super().fit(X, y, epochs, verbose, validation_data)

