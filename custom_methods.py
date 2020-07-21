# Custom methods to clean up the source code a bit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class cust():
    
    def __init__(self, dat):
        self.dat = dat

        
    def xy(self, x_dummies_list, X_list, y_list):
        
        X = self.dat[X_list]
        y = self.dat[y_list]
        
        # One hot encoding the pg column
        X = pd.get_dummies(X, columns = x_dummies_list, drop_first = True)
        # C_cols should be after get_dummies so we get ALL columns
        x_cols = X.columns
        # Convert the two into value arrays
        X = X.values
        y = y.values
        # We need y as a 1D array
        y = np.ravel(y)

        return X,y,x_cols
    
    
    # Same as xy except returns a dataframe instead of a float64 np array
    def xy_df(self, x_dummies_list, X_list, y_list):
        
        X = self.dat[X_list]
        y = self.dat[y_list]
        
        # One hot encoding the pg column
        X = pd.get_dummies(X, columns = x_dummies_list, drop_first = True)
        # C_cols should be after get_dummies so we get ALL columns
        x_cols = X.columns
        # We need y as a 1D array
        y = np.ravel(y)

        return X,y,x_cols    
    
    
        # var Specific
    def pg_filter(self, pg, X_list):
        filtered = self.dat[X_list]
        filtered = filtered[filtered['pg'] == pg]
        plt.plot(filtered['ppg'])
        return filtered
    
    def clean_data(self):
        self.dat.replace([np.inf, -np.inf], np.nan) # Replace inf
        self.dat = self.dat.dropna(axis=0, how = 'any') # Drop NA's on the rows axis
        # I kept getting a value error and this was the only thing that seemed to fix it
        self.dat = self.dat[~self.dat.isin([np.nan, np.inf, -np.inf]).any(1)]
        return self.dat
    
    
    def outlier_removal(self,var):
        IQR = self.dat[var].describe()['75%'] - self.dat[var].describe()['25%']
        min_val = self.dat[var].describe()['25%'] - (IQR * 1.5)
        max_val = self.dat[var].describe()['75%'] + (IQR * 1.5)
        
        self.dat = self.dat[(self.dat[var] > min_val) & (self.dat[var] < max_val)]
        plt.boxplot(self.dat[var])
        return self.dat
         
    @staticmethod
    def comparison_df(y_pred, y_test):
        # Dataframe of pred and actual y
        comparison_df = pd.DataFrame({'y_pred':y_pred, 'y_test':y_test})
        comparison_df['abs_difference'] = abs( comparison_df['y_pred'] - comparison_df['y_test'] )
        comparison_df['real_difference'] = comparison_df['y_pred'] - comparison_df['y_test'] 
        print(comparison_df.describe())
        # Show all sums
        print(comparison_df.sum())
        # Show average difference
        print ("Average Difference: ", comparison_df.sum()[2] / len(comparison_df))
        
        return comparison_df

    @staticmethod
    def state_df(dat, list_of_states):
        dat_backup = dat.copy()
        dat = dat[dat['state'].isin(list_of_states)]
        return dat_backup, dat
        
