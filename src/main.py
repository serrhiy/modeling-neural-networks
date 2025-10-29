import numpy
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, SimpleRNN, Reshape

RANGE = 10
POINTS_NUMBER = 1000


def get_learning_set() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    x = numpy.linspace(-RANGE, RANGE, POINTS_NUMBER)
    y = numpy.sin(x) + numpy.cos(x / 2)
    z = numpy.sin(2 * numpy.sqrt(x**2 + y**2)) / (numpy.sqrt(x**2 + y**2) + 0.001)
    return (x, y, z)


def test_model(model, learning_set: tuple[numpy.ndarray], epochs=20, batch_size=100):
    x, y, z = learning_set
    inputs = numpy.stack([x, y], axis=1)

    model.compile(optimizer="adam", loss="mse")
    model.fit(inputs, z, epochs=epochs, batch_size=batch_size, verbose=0)

    predicted = model.predict(inputs)

    mse = numpy.mean((z - predicted) ** 2)
    print(f"Mean Squared Error: {mse}")

    pyplot.figure(figsize=(8, 5))
    pyplot.plot(x, z, label="Actual", color="blue")
    pyplot.plot(x, predicted, label="Predicted", color="orange")
    pyplot.title("Model Prediction vs Actual")
    pyplot.xlabel("X value")
    pyplot.ylabel("Output value")
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()


def create_feed_forward_model(layers: int, neurons: int):
    model = Sequential()
    model.add(Input(shape=(2,)))
    for _ in range(layers):
        model.add(Dense(neurons, activation="relu"))
    model.add(Dense(1, name="output"))
    return model


def create_cascade_forward_model(layers: int, neurons: int):
    input_layer = Input(shape=(2,))
    current = Dense(neurons, activation="relu")(input_layer)
    for _ in range(layers - 1):
        concatenated_layer = Concatenate()([input_layer, current])
        current = Dense(neurons, activation="relu")(concatenated_layer)
    output_layer = Dense(1, name="output")(current)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def create_elman_backprop(layers: int, neurons: int):
    model = Sequential()
    model.add(Reshape((1, 2), input_shape=(2,), name="input_reshape"))
    for _ in range(layers - 1):
        model.add(SimpleRNN(neurons, return_sequences=True, activation="relu"))
    model.add(SimpleRNN(neurons, return_sequences=False, activation="relu"))
    model.add(Dense(1, name="output"))
    return model


def main():
    learning_set = get_learning_set()

    m1 = create_feed_forward_model(1, 10)
    m2 = create_feed_forward_model(1, 20)
    test_model(m1, learning_set)
    test_model(m2, learning_set)

    m3 = create_cascade_forward_model(1, 20)
    m4 = create_cascade_forward_model(2, 10)
    test_model(m3, learning_set)
    test_model(m4, learning_set)

    m5 = create_elman_backprop(1, 15)
    m6 = create_elman_backprop(3, 5)
    test_model(m5, learning_set)
    test_model(m6, learning_set)


if __name__ == "__main__":
    main()
