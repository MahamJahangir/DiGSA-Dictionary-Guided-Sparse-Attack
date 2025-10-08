import tensorflow as tf

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D(strides=2))
    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.build()
    model.summary()
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test, batch_size=100, epochs=30):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    return history

def save_and_load_weights(model, checkpoint_path='my_checkpoint'):
    model.save_weights(checkpoint_path)
    model.load_weights(checkpoint_path)
