rnn_df = rnn_df.sort_index(ascending = True)
rnn_df = rnn_df.reset_index()

# Create the initial results df with a look_back of 60 days
result = []
sequence_length = 60

# 3D Array
for index in range(len(rnn_df) - sequence_length):
    result.append(rnn_df['gal'][index: index + sequence_length])  

# Getting the initial train_test split for our min/max val scalar
train_test_split = 0.9
row = int(round(train_test_split * np.array(result).shape[0]))
train = np.array(result)[:row, :]
X_train = train[:, :-1]


# Manual MinMax Scaler
X_min = X_train.min()
X_max = X_train.max()

# Minmax scaler and a reverse method
def minmax(X):
    return (X-X_min) / (X_max - X_min)

def reverse_minmax(X):
    return X * (X_max-X_min) + X_min

# Method for Scaler for each window in our 3D array
def minmax_windows(window_data):
    normalised_data = []
    for window in window_data:
        window.index = range(sequence_length)
        normalised_window = [((minmax(p))) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

# minmax the windows
result = minmax_windows(result)
# Convert to 2D array
result = np.array(result)




# Train/test for real this time
row = round(train_test_split * result.shape[0])
train = result[:row, :]

# Get the sets
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = result[row:, :-1]
y_test = result[row:, -1]

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1,1))
y_test = np.reshape(y_test, (-1,1))
