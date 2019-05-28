# Gaussian Process Regression for Predicting Stock Prices

<p>This project illustrates how to use Gaussian Process to predict Stock Markets. We specifically stocks of Google, Netflix and GE as examples in this case.</p>

# Idea: 
<p> We are going to deal with two forecasting problems, one is predict yearly, another is quarterly. <p>
1. Independent variables (X): the year and the day of the year
2. Dependent variable (Y): normalized adjusted closing price for each day in a year
3. We're going to build a new df, with each row is one day, each column in one year, and the num_rows is limitted to 252 as common knowledge
4. We also add the fiscal quarter associated with each row to predict for quarterly stock
5. make_gp_predictions contains pred_quarters param to indicate whether predicting quarters specified instead of the entire year.
  E.g: pre_quaters = [4] means the function will predict for quarter 4 of  2018, using all the data till quarter 3 of 2018

### Installations
1. This code was checked on Python 3.7.3 (Anaconda3)
2. Create a conda virtual environment and install packages using requirements.txt by conda and pip


### Python Code Run Instructions
Run main.py to execute the entire code
```
python main.py
```

#### Dataset
The dataset was downloaded from [Yahoo Finance](https://finance.yahoo.com). We downloaded the entire stock history for three companies:
1. [Google] (https://finance.yahoo.com/quote/GOOG)
2. [Netflix] (https://finance.yahoo.com/quote/NFLX)
3. [General Electric Company] (https://finance.yahoo.com/quote/GE)  

### Dataset Description
1. Date: calendar date when the price of the stock was measured
2. Open: the opening price of the day
3. High: the highest price of the day
4. Low: the lowest price of the day
5. Close: the closing price of the day
6. Adj Close: the adjusted closing price, which is also our target variable Y in the dataset
7. Volume: number of shares traded during a day 


### Code Details
1. main.py :  Main function which runs the entire code
2. PreProcessing.py :  Preprocesses the stock data to make it ready for modeling
3. VisualizeData.py : Contains the functions to visualize the dataset
4. GP.py : Contains the implementation of training and inference through Gaussian Process using GpFlow library


