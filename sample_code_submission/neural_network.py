# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# class NeuralNetwork:
#     """
#     This class implements a neural network classifier with BatchNormalization and Dropout.
#     """

#     def __init__(self, train_data):
#         self.model = Sequential()

#         n_dim = train_data.shape[1]

#         self.model.add(Dense(64, input_dim=n_dim))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(64, activation="relu"))
#         self.model.add(Dropout(0.3))

#         self.model.add(Dense(32))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(32, activation="relu"))
#         self.model.add(Dropout(0.3))

#         self.model.add(Dense(16))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(16, activation="relu"))
#         self.model.add(Dropout(0.2))

#         self.model.add(Dense(1, activation="sigmoid"))  # Binary classification

#         self.model.compile(
#             loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
#         )

#         self.scaler = StandardScaler()

#     def fit(self, train_data, y_train, weights_train=None):
#         # Callbacks
#         early_stop = EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         )

#         reduce_lr = ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=2,
#             min_lr=1e-6,
#             verbose=1
#         )

#         self.scaler.fit(train_data)
#         X_train = self.scaler.transform(train_data)

#         self.model.fit(
#             X_train,
#             y_train,
#             sample_weight=weights_train,
#             validation_split=0.2,
#             epochs=20,
#             batch_size=32,
#             verbose=2,
#             callbacks=[early_stop, reduce_lr]
#         )

#     def predict(self, test_data):
#         test_data = self.scaler.transform(test_data)
#         return self.model.predict(test_data).flatten().ravel()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(64, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))  


        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):
        
        # Early stopping 
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2 , callable=[early_stop]
        )

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()

