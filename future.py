# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Predicting the future
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from numpy import newaxis
# We sorted descending
curr_frame = X_test[len(X_test)-1]
future = []

# Quick plot of the frame we're predicting from
plt.plot(curr_frame)

points_to_predict = 10
for i in range(points_to_predict):
      # append the prediction to our empty future list
     future.append(model.predict(curr_frame[newaxis,:,:])[0,0])
      # insert our predicted point to our current frame
     curr_frame = np.insert(curr_frame, len(X_test[0]), future[-1], axis=0)
      # push the frame up one to make it progress into the future
     curr_frame = curr_frame[1:]
     
plt.plot(future)    
