import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # original dataset
    data = np.genfromtxt('../dataset/processed_insurance.csv', delimiter=',', skip_header=1)
    X = data[:, 0:6]
    Y = data[:, 6]
    print(X.shape)
    print(Y.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(512, activation='relu'),
        Dense(1)  # Linear activation for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.1)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")
    print([y_test[1], y_pred[1][0]])
