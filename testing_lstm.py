
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
    plt.ylabel('Volume')
    plt.xlabel('Observation #')
    plt.legend(['Predicted', 'Actual'], loc='upper right')
    plt.show()

# plot the test set vs real    
hist_plt(predictions , y_test)


# Unscale
predictions_unscaled = reverse_minmax(predictions)
actuals_unscaled = reverse_minmax(y_test)
hist_plt(predictions_unscaled , actuals_unscaled)


comparison_df = pd.DataFrame({"Real": actuals_unscaled, "Pred": predictions_unscaled}, index = range(len(predictions_unscaled)))
# Print out results
print("Mean of the actuals is: {}\nMean of the predicted is: {}".format(round(comparison_df.mean()[0],3), round(comparison_df.mean()[1],3)))
print("Sum of the actuals is: {}\nSum of the predicted is: {}".format(round(comparison_df.sum()[0],4), round(comparison_df.sum()[1]),4))
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
