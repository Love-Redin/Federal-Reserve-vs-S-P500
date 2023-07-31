#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:48:49 2022

Author: Love Redin
"""

#TODO: add animation/gradual color difference to show the time series order of the data points

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    """
    Main function.
    
    Reads data, does a linear regression between total Fed assets and S&P500 price, then plots the result.
    
    """
    fed_df, fed_dates = read_fed_data("fed_data.txt")
    sp_df, sp_dates = read_sp_data("sp_data.csv")
    
    fed_df = remove_missing_dates(fed_df, sp_dates)
    sp_df = remove_missing_dates(sp_df, fed_dates)
        
    # Check that all data has the same length
    assert(len(fed_df) == len(sp_df) == len(sp_df["Date"]) == len(fed_df["Date"]))
    
    # Join the datasets together
    df = pd.concat([sp_df.set_index('Date'), fed_df.set_index('Date')], axis=1, join='inner')
    
    first_year = df.index.min()[:4]
    last_year = df.index.max()[:4]

    X = df["Total Assets"] # Federal Reserve balance sheet
    Y = df["Close"] # S&P500 closing price

    fig, ax = plt.subplots()
       
    df.plot(kind="scatter", x="Total Assets", y="Close", s=2, ax=ax, color="b")
    
    ax.set_xlabel("Total assets of the Federal Reserve [trillions of USD]")
    ax.set_ylabel("S&P500 [USD]")
    ax.set_title(f"The S&P500 as a function of the Fed's total assets between {first_year} and {last_year}", wrap=True)
    
    X = X.values.reshape(-1, 1)
    Y = Y.values.reshape(-1, 1)  
    
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.plot(X, Y_pred, color='red')
    
    slope = round(linear_regressor.coef_[0][0])
    intercept = round(linear_regressor.intercept_[0])
    
    print("Slope:", slope)
    print("Intercept:", intercept)
    
    ax.text(1, 4500, f'$Y = {slope} * X + {intercept}$', fontsize=10)
    
    #plt.savefig('fed_vs_sp.eps', format='eps')

    plt.show()

    
def read_fed_data(filename):
    """
    Reads and cleans Federal Reserve Balance Sheet data.
    
    """
    fed_df = pd.read_csv(filename, sep="\t", header=None)
    first_row = fed_df.iloc[0, :]
    fed_df.columns = first_row
    fed_df = fed_df.drop(0)
    fed_df['Date'] = fed_df['Date'].apply(transform_date)
    fed_dates = fed_df["Date"]
    
    # Convert from millions to trillions
    fed_df["Total Assets"] = pd.to_numeric(fed_df["Total Assets"])/10**6

    return fed_df, fed_dates
    
def read_sp_data(filename):
    """
    Reads and cleans S&P500 price data.
    
    """
    sp_df = pd.read_csv(filename, sep=";")
    
    # We only need columns Date and Close (=closing price)
    sp_df = sp_df[["Date", "Close"]]
    sp_df['Date'] = sp_df['Date'].apply(transform_date)
        
    # Reverse order of dates, giving us the oldest first
    sp_df = sp_df.reindex(index=sp_df.index[::-1]).reset_index(drop=True)

    sp_dates = sp_df["Date"]
    
    return sp_df, sp_dates
    

def transform_date(date):
    """
    Transforms a date from any format to the format YYYY-MM-DD.
    """
    new_date = pd.to_datetime(date).strftime('%Y-%m-%d')
    return new_date


def remove_missing_dates(df, dates):
    """
    Cleans a Pandas dataframe based on the column Date, removing dates which do not have an exact match in the S&P500 data.
    This makes the linear regression possible.
    Returns the dataframe with unmatched dates removed.
    """
    missing_dates = [index for index, date in enumerate(df.Date) if date not in dates.values]
    df = df.drop(missing_dates, axis=0).reset_index(drop=True)
    return df

    
if __name__ == '__main__':
    main()
    
