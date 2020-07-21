
# ~~~~~~~~~~~~~~~~~~ TRAINING THE RNN
from keras.models import Sequential
from keras.layers import LSTM, Dense
EPOCHS = 100

# 2 layers + 1 output layer, 100 nodes each, using stochastic gradient descent and mean squared error loss
model = Sequential()

# Add the first layer.... the input shape is (Sample, seq_len-1, 1). Return_sequences so we can add a second layer
model.add(LSTM(
        input_shape = (sequence_length-1, 1), return_sequences = True,
        units = 100))

# Add the second layer.... the input shape is (Sample, seq_len-1, 1)
model.add(LSTM(
        input_shape = (sequence_length-1, 1), 
        units = 100))

# Add the output layer, simply one unit
model.add(Dense(
        units = 1,
        activation = 'linear'))

model.compile(loss = 'mse', optimizer = 'adam')


model.fit(X_train, y_train, epochs = EPOCHS, validation_split = 0.05)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Testing the Test set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Predict X_Test and reshape to 1D for dataframe
predictions = model.predict(X_test)
predictions = np.reshape(predictions, (predictions.size,))

# Reshape y_test to 1D for dataframe
y_test = np.reshape(y_test, (y_test.size,))

def hist_plt(pred, y):
    plt.plot(pred)
    plt.plot(y)
    plt.title('Analysis of Predictions vs Test Data')
    plt.ylabel('Gallons')
    plt.xlabel('Observation #')
    plt.legend(['Predicted', 'Actual'], loc='upper right')
    plt.show()

# plot the test set vs real    
hist_plt(predictions , y_test)

comparison_df = pd.DataFrame({"Real": y_test, "Pred": predictions}, index = range(len(predictions)))
# Print out results
print("Mean of the actuals is: {}\nMean of the predicted is: {}".format(round(comparison_df.mean()[0],3), round(comparison_df.mean()[1]),3))
print("Sum of the actuals is: {}\nSum of the predicted is: {}".format(round(comparison_df.sum()[0],4), round(comparison_df.sum()[1]),4))

# Unscale
predictions_unscaled = reverse_minmax(predictions)
actuals_unscaled = reverse_minmax(y_test)
hist_plt(predictions_unscaled , actuals_unscaled)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Testing the Training set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plot the train set vs real (to see if it's overtrained)
# Predict X_train
predictions_train = model.predict(X_train)
predictions_train = np.reshape(predictions_train, (predictions_train.size,))

# Get our  y_train and make 1D for df
y_train = np.reshape(y_train, (y_train.size,))

# Plot it
hist_plt(predictions_train, y_train)

comparison_df_train = pd.DataFrame({"Real_train": y_train, "Pred_train": predictions_train},  index = range(len(predictions_train)))
print("Mean of the actuals is: {}\nMean of the predicted is: {}".format(round(comparison_df_train.mean()[0],3), round(comparison_df_train.mean()[1]),3))
print("Sum of the actuals is: {}\nSum of the predicted is: {}".format(round(comparison_df_train.sum()[0],4), round(comparison_df_train.sum()[1]),4))
