"""
Description of this stock predictor:
This is my Investor's 8ball. It focuses on using news headlines, previous stock prices, and machine
learning to calculate the price of a stock on any date. Currently, we are starting off with an sklearn
framework for linear regression, which will be used to predict the stock prices, and then, because
numbers are not enough to express what happens in the real world, we will use news headlines and Reddit and Twitter
posts to calculate a weight, which we will multiply each of the stock predictions by to get our final result.

What is Linear Regression?
Linear regression is a very basic algorithim that essentially looks at a scatter of data points and calculates a 
best-fit line; basically a line that can be used to predict the output value(s) given some input value(s). Linear 
regression is used when the data correlates to itself. The line is drawn using a loss function and a certain number
of dimensions (attributes). Then the line drawn will be used to predict the labels.

Some Important Information:
-Attributes are the input data, they can also be called features
-Labels are the output data, they can also be called targets

By: Karthik Singaravadivelan
"""

# All the imports go below
# These modules are for yahoo finance information purposes and news scraping
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

# My usual machine learning imports:
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpre
import numpy as np
import pandas as pd

# These are for visualization
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

# All regressors that I may need
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR

# These imports are for evaluating different models
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

# This import statement includes something I need to compare models
from sklearn.pipeline import make_pipeline

# imports included with python that I need
import os
import sys
import time
import pickle
import random
import datetime
import re
import json
import csv
import math
from io import StringIO
from pprint import pprint

# Tutorial on how to get stock data:
# https://www.youtube.com/watch?v=fw4gK-leExw

# These videos show how the stock market works:
# How do investors choose stocks? - Richard Coffin -----> https://www.youtube.com/watch?v=CMQLdJa64Wk
# How does the stock market work? - Oliver Elfenbaum ---> https://www.youtube.com/watch?v=p7HKvqRI_Bo

# This is a quick function to write to a text file
def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

# this function clears the log file
def clear_log():
    file = open("log.txt","r+")
    file.truncate(0)
    file.close()

# This function gets a model's name (from scikit-learn ONLY)
def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

# STOCK PREDICTOR CLASS BELOW

class StockPredictor(object): # v2
    # The docstring is below:
    """
    My Stock Predictor class (v2)

    By Karthik Singaravadivelan

    This class predicts stocks. There is no dataset needed, and there are very little commands involved.
    
    First, you must create an instance of this class.
    Then, use the load_data() method to load your dataset for the predicting.
    After you've loaded your data, use the fit method and then the predict method to predict your stock prices.

    Note: The accuracy of this model ranges from 0.8 to 0.98 based on the attributes and data given.

    The Best Model for this class is KNeighborsRegressor(n_neighbors=5), but if you want a different model, 
    feel free to set your own with the self.model parameter. Also, try out the "pick_model" function to pick
    a model with the best RMSE. This will not always generate the best results because the best RMSE has 
    nothing to do with the real world.

    """
    # the constructor
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker.upper()
        self.attributes = ["Date","Open","High","Low","Close","Adj Close","YesterdayOpen","YesterdayHigh","YesterdayLow","YesterdayClose","YesterdayAdj Close", "ShortEMA", "MiddleEMA", "LongEMA"]
        self.labels = ["Open","High","Low","Close","Adj Close"]
        self.model = Lasso(tol=0.002) # a variable to hold the regression model.
        self.model_path = "stockmodel.pickle"
        self.data = None # a variable to hold the data python object.
        self.dataset = R"data\{}.csv".format(str(self.stock_ticker))# actual data
        self.future_dataset = R"data\Future{}.csv".format(str(self.stock_ticker))# dataset for predictions
        self.test_attrs = [x for x in self.attributes if x not in self.labels] # all the attributes for testing the model
        self.num_test_attrs = len(self.test_attrs)
        self.all_stmts = None # a variable to hold all the business statements
        self.moving_avgs = [5, 21, 63]

    def load_data(self):
        """
        This function will load the data from the Yahoo Finance website with BeautifulSoup4
        and put it in a csv file.
        """
        url_stats = f'https://finance.yahoo.com/quote/{self.stock_ticker}/key-statistics?p={self.stock_ticker}'
        url_profile = f'https://finance.yahoo.com/quote/{self.stock_ticker}/profile?p={self.stock_ticker}'
        url_financials = f'https://finance.yahoo.com/quote/{self.stock_ticker}/financials?p={self.stock_ticker}'

        dataset_filename = self.dataset

        if os.path.exists(dataset_filename):
            os.remove(dataset_filename)

        if os.path.exists(self.future_dataset):
            os.remove(self.future_dataset)

        if os.path.exists(self.model_path):
            os.remove(self.model_path)

        response = requests.get(url_financials)
        soup = BeautifulSoup(response.text, 'html.parser')

        pattern = re.compile(r'\s--\sData\s--\s')
        script_data = soup.find('script', text=pattern).contents[0]

        start = script_data.find("context")-2
        end = -12

        json_data = json.loads(script_data[start:end])
        # json_data['context'].keys()

        # IS means Income Statement
        annual_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['incomeStatementHistory']['incomeStatementHistory']
        quarterly_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['incomeStatementHistoryQuarterly']['incomeStatementHistory']

        # CF means Cash Flow
        annual_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['cashflowStatementHistory']['cashflowStatements']
        quarterly_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['cashflowStatementHistoryQuarterly']['cashflowStatements']

        # BS means Balance Sheet
        annual_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['balanceSheetHistory']['balanceSheetStatements']
        quarterly_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['balanceSheetHistoryQuarterly']['balanceSheetStatements']

        # This function extracts statements from the json data
        def extract_statements(statement_json):
            stmts = []
            for s in statement_json:
                statement = {}
                for key,val in s.items():
                    try:
                        statement[key] = val['raw']
                    except TypeError:
                        continue
                    except KeyError:
                        continue
                stmts.append(statement)
            return stmts

        # Extracting all the statements with my function
        annual_is_stmts = extract_statements(annual_is)
        quarterly_is_stmts = extract_statements(quarterly_is)

        annual_cf_stmts = extract_statements(annual_cf)
        quarterly_cf_stmts = extract_statements(quarterly_cf)
        
        annual_bs_stmts = extract_statements(annual_bs)
        quarterly_bs_stmts = extract_statements(quarterly_bs)

        # Compiling all the statements into a list
        # IS means Income Statement
        # CF means Cash Flow
        # BS means Balance Sheet
        self.all_stmts = [annual_is_stmts, quarterly_is_stmts, annual_cf_stmts, quarterly_cf_stmts, annual_bs_stmts, quarterly_bs_stmts]

        # Profile Data
        response = requests.get(url_profile)
        soup = BeautifulSoup(response.text, 'html.parser')

        pattern = re.compile(r'\s--\sData\s--\s')
        script_data = soup.find('script', text=pattern).contents[0]

        start = script_data.find("context")-2
        end = -12

        json_data = json.loads(script_data[start:end])

        business_summary = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['assetProfile']['longBusinessSummary']

        # Time for the stats!
        response = requests.get(url_stats)
        soup = BeautifulSoup(response.text, 'html.parser')

        pattern = re.compile(r'\s--\sData\s--\s')
        script_data = soup.find('script', text=pattern).contents[0]

        start = script_data.find("context")-2
        end = -12

        json_data = json.loads(script_data[start:end])

        stats = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['defaultKeyStatistics']

        ################################
        # Historical Stock Data

        stock_url = f'https://query1.finance.yahoo.com/v7/finance/download/{self.stock_ticker}?'
        
        params = {
            'range':'20y',
            'interval':'1d',
            'events':'history'
        }
        response = requests.get(stock_url, params=params)

        file = StringIO(response.text)
        reader = csv.reader(file)
        csvdata = list(reader)

        # turning everything into a csv:
        # this formats everthing along the way as well.
        # Basically what this loop does is, instead of just writing a list to the end of the 
        # csv file, this is turning everthing into floats and integers
        # so the python script can read everything properly.
        for listrow in range(len(csvdata)):
            row = csvdata[listrow]
            newrow = []
            if listrow >= 1:
                for thing in row:
                    if row.index(thing) == 6:
                        continue
                    try:
                        thing = float(thing)
                        newrow.append(thing)

                    except ValueError:
                        date = thing
                        newrow.append(listrow)
                if listrow >= 2:
                    newprevrow = []
                    prevrow = csvdata[listrow - 1]
                    prevrow.pop(0)
                    for thing in prevrow:
                        if prevrow.index(thing) == 5:
                            continue
                        else:
                            thing = float(thing)
                            newprevrow.append(thing)
                else:
                    prevrow = [0, 0, 0, 0, 0]
                    newprevrow = prevrow
                
                newrow.extend(newprevrow)

                newrow.append(date)

            else:
                # this block of code recognizes if the line they are trying to convert to csv are the headings/attributes
                if "{" in row:
                    continue
                newrow = row
                other_headings = []
                other_headings.extend(self.test_attrs)
                other_headings.remove("Date")
                other_headings.append("RealDate")
                newrow.remove("Volume")
                newrow.extend(other_headings)

            # Finally, writing the csv file
            with open(dataset_filename, 'a', newline="") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                today = str(datetime.datetime.now().strftime("%Y-%m-%d"))
                future = newrow[-1]
                condition_of_date = future == today
                if not condition_of_date:
                    writer.writerow(newrow)
            

        # Some weirdo Preprocessing
        df = pd.read_csv(self.dataset)
        df["RealDate"] = df['ShortEMA']
        shortEMA = df.Close.ewm(span=self.moving_avgs[0], adjust=False).mean()
        middleEMA = df.Close.ewm(span=self.moving_avgs[1], adjust=False).mean()
        longEMA = df.Close.ewm(span=self.moving_avgs[2], adjust=False).mean()

        df['ShortEMA'] = shortEMA
        df['MiddleEMA'] = middleEMA
        df['LongEMA'] = longEMA

        df.to_csv(self.dataset, index=False)

        os.popen(f'copy {self.dataset} {self.future_dataset}')

        """
        The label meanings:
        Open - The stock price when the market opens.
        Close - The stock price when the market closes.
        High - The highest stock price of the day.
        Low - The lowest stock price of the day.
        Adj Close - the closing price after adjustments for all applicable splits and dividend distributions.
        Short through Long EMAs - Moving averages calculated for different periods of time.
        """

    def pick_model(self, show=False, test_size=0.1):

        """
        This method will automatically select the best model for you based on r2 scores and root-mean-squared error. 
        Note that this is not always reliable as RMSE and R2 have no real-world input other than the numbers.
        """
        
        data = pd.read_csv(self.dataset, sep=",")
        data = data[self.attributes]
        self.data = data
        predict = self.labels

        X = np.array(data.drop(predict, 1)) # ftrs
        y = np.array(data[predict]) # tgt
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # print("\nTraining set: {} samples".format(X_train.shape))
        # print("Test set: {} samples\n".format(X_test.shape))

        old_school = [
            LinearRegression(),
            KNeighborsRegressor(n_neighbors=3),
            KNeighborsRegressor(n_neighbors=9)
        ]

        penalized_lr = [Lasso(tol=0.002), Ridge()]

        dtrees = [DecisionTreeRegressor(max_depth=md) for md in [1, 3, 5, 10]]

        reg_models = old_school + penalized_lr + dtrees

        def rms_error(actual, predicted):
            mse = mean_squared_error(actual, predicted)
            return np.sqrt(mse)

        scores = {}
        for model in reg_models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            key = (get_model_name(model) + 
            str(model.get_params().get('max_depth', "")) +
            str(model.get_params().get('n_neighbors', "")))
            scores[key] = [rms_error(y_test, preds), model]

        df = pd.DataFrame.from_dict(scores, orient='index').sort_values(0)
        df.columns = ['RMSE', 'MODEL_CLASS']

        dfd = df.to_dict()
        best_model = min(dfd["RMSE"], key=dfd["RMSE"].get)
        self.model = dfd['MODEL_CLASS'][str(best_model)]

        fig = plt.figure(figsize=(128, 64))
        rmses = [value for (key, value) in dfd['RMSE'].items()]
        names = list(df.index)
        plt.bar(names, rmses, color='white')
        plt.plot(range(len(names)), rmses, color='red')
        plt.scatter(range(len(names)), rmses, color='black')
        
        # Some styling
        plt.xticks(np.arange(len(names)), names)
        plt.ylabel('Root Mean Squared Error')

        if show:
            plt.show()

        return df

    def fit_inital(self, filename=None):
        """
        Here we are fitting our first model on the original data:
        """
        if not filename:
            filename = self.dataset

        # Clearing the log
        clear_log()

        append_new_line("log.txt", "\nFitting the model for the stock predictor. \nStarting log at "+str(datetime.datetime.now())+"\n")

        data = pd.read_csv(filename, sep=",")
        data = data[self.attributes]

        self.data = data

        predict = self.labels

        # print(num_test_attrs, test_attrs)

        X = np.array(data.drop(predict, 1))
        y = np.array(data[predict])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        append_new_line("log.txt", "\nTraining set: {} samples".format(X_train.shape))
        append_new_line("log.txt", "Test set: {} samples".format(X_test.shape))

        try:
            best = pickle.load(open(self.model_path, 'rb'))[1]
        except FileNotFoundError:
            best = 0

        if best != 0:
            if pickle.load(open(self.model_path, 'rb'))[0] != self.model:
                # print("Old Model:", pickle.load(open(self.model_path, 'rb'))[0])
                # print("New Model:", self.model)
                os.remove(self.model_path)
                best = 0
        
        num_times = 30
        for _ in range(num_times):

            linear = self.model

            linear.fit(X_train, y_train)

            acc = linear.score(X_test, y_test)
            # print("Accuracy: "+str(acc))

            if acc > best:
                with open(self.model_path, 'wb') as f:
                    pickle.dump([linear, acc], f)
                
                best = acc
        
        best_acc_str = "\nBest Accuracy: "+str(best)+"\n"
        # print(best_acc_str)
        pickle_in = open(self.model_path, 'rb')

        linear = pickle.load(pickle_in)[0]
        self.model = linear

        append_new_line("log.txt", "\nFitting Finished at "+str(datetime.datetime.now()))

        return best

    def predict(self, date, filename=None):
        """
        This predictor method predicts stocks, given a date in the form of YYYY-MM-DD.
        It will predict any date, however the farther into the future you choose, the longer it will take to predict the stock price, and the less reliable the machine will be.
        """
        

        # What this model is doing is, it's predicting one day into the future, 
        # then it adds the values it gets to the Future(stock_ticker).csv dataset.
        # Then it trains on the new values, and predicts one more day into the future.
        # This process continues until the date predicted is the date the user asks for, 
        # where the console reveals the predicted values.

        # We need to define the year, month, and day to find out how many times the loop should be ran.

        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])

        # print("Y:{}\nM:{}\nD:{}\n".format(year, month, day))
        
        def numOfDays(date1, date2):
            return (date2-date1).days

        # This function can read any cell in a csv file.
        def read_cell(x, y):
            filename = self.future_dataset
            with open(filename, 'r') as f:
                rows = list(csv.reader(f))
                cell = rows[y][x]
                return cell

        def calculate_EMA(close_today, EMA_yester, num_days):
            EMA = (close_today * (2 / (num_days - 1)) + EMA_yester * (1 - (2 / (num_days - 1))))
            return EMA

        print("Model Used For Prediction: " + str(get_model_name(self.model)))
        # This function is to see how many times the "for" loop should run.
        today = datetime.date.today()
        today = today - datetime.timedelta(days=1)
        date1 = datetime.date(year, month, day)
        date2 = datetime.date(today.year, today.month, today.day)
        # print(today.year, today.month, today.day)
        days = numOfDays(date2, date1)
        # print(days, "days")

        stock_days = days + float(read_cell(0, -1)) - 1

        for day in range(0, days):

            # print(day)

            """
            Here we are fitting our first model:
            """
            if not filename:
                filename = self.future_dataset

            append_new_line("log.txt", "\nFitting the model for the stock predictor. \nStarting log at "+str(datetime.datetime.now())+"\n")

            data = pd.read_csv(filename, sep=",")
            data = data[self.attributes]

            predict = self.labels

            # print(num_test_attrs, test_attrs)

            X = np.array(data.drop(predict, 1))
            y = np.array(data[predict])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            # print ("\n\nTraining set: {} samples".format(X_train.shape))
            # print ("Test set: {} samples".format(X_test.shape))

            try:
                best = pickle.load(open(self.model_path, 'rb'))[1]
            except FileNotFoundError:
                best = 0
            
            num_times = 30
            for _ in range(num_times):

                linear = self.model

                linear.fit(X_train, y_train)

                acc = linear.score(X_test, y_test)
                # print("Accuracy: "+str(acc))

                if acc > best:
                    with open(self.model_path, 'wb') as f:
                        pickle.dump([linear, acc], f)
                    
                    best = acc
            
            best_acc_str = "\nBest Accuracy: "+str(best)+"\n"
            # print(best_acc_str)
            pickle_in = open(self.model_path, 'rb')

            linear = pickle.load(pickle_in)[0]
            self.model = linear
            # print(coef_str)
            # print(y_in_str)

            append_new_line("log.txt", "\nPredictions Fitting Finished at "+str(datetime.datetime.now()))

            """
            Here is where the predicting begins.
            """
            current_sd = stock_days - (days - day - 1)

            pred_values = [current_sd]
            # print(days, stock_days)
            
            for pred_attr in self.labels:
                cell_x = int(self.attributes.index(pred_attr))
                cell_y = -1
                # print(f"({cell_x}, {cell_y})")
                pred_value = read_cell(cell_x, cell_y)
                pred_value = float(pred_value)

                pred_values.append(pred_value)

            # print(pred_values)

            my_values = np.array([pred_values])[0]
            new_short_EMA = calculate_EMA(
                float(read_cell(4, -1)), float(read_cell(11, -1)), self.moving_avgs[0])
            new_middle_EMA = calculate_EMA(
                float(read_cell(4, -1)), float(read_cell(12, -1)), self.moving_avgs[1])
            new_long_EMA = calculate_EMA(
                float(read_cell(4, -1)), float(read_cell(13, -1)), self.moving_avgs[2])

            my_values = np.append(my_values, [new_short_EMA, new_middle_EMA, new_long_EMA])
            my_values = my_values.reshape(-1, self.num_test_attrs)

            formatted_pred_values = []
            for value in my_values.tolist()[0]:
                value = round(float(value), 2)
                formatted_pred_values.append(value)

            prediction = linear.predict(my_values)
            output_values = prediction[0]

            formatted_output_values = []
            for value in output_values:
                value = round(float(value), 2)
                formatted_output_values.append(value)

            attr_d = {"Input Labels":self.test_attrs, "Input Values":[float(round(x, 2)) for x in formatted_pred_values]}
            attr_df = pd.DataFrame(data=attr_d)
            label_d = {"Output Labels":self.labels, "Output Values":[float(round(x, 2)) for x in formatted_output_values]}
            label_df = pd.DataFrame(data=label_d)

            df_str = str(attr_df) + "\n\n" + "Predicted Date: {}\n\n".format(date) + str(label_df)
            append_new_line("log.txt", df_str)

            if day == days-1:
                # print("\n")
                # print(df_str)
                pass

            future_data = []
            future_data.extend(pred_values)
            future_data[0] = int(future_data[0]) + 1
            for elem in reversed(list(output_values)):
                future_data.insert(1, round(elem, 5))
            future_data.extend([new_short_EMA, new_middle_EMA, new_long_EMA])
            # print("\nData to be plugged into Future Dataset: {}".format(future_data))

            future_data[0] = int(future_data[0])

            # print(future_data)

            last_date = stock_days - days + 1 # This variable is the last real-value date predicted by the object

            current_stock_date = date

            future_data.append(current_stock_date)

        print("\nSee log.txt for more details about the training and testing of the model. (And all the dataframes)")
        # Writing to the Excel file: predictions.xlsx
        wb_name = 'data\\predictions.xlsx'
        try:
            label_df.to_excel(wb_name)
            return label_df, attr_df, date
        except UnboundLocalError:
            print("You have predicted a date in the past!!")
            exit()

    def weight_preds(self, preds=None):
        """
        This function will add weights to the predictions given by the 'predict' method and will return
        newly produced, weighted outputs that will be of higher accuracy.
        """

        # First we need to get all the available articles
        get_articles(self.stock_ticker)

        def read_cell(x, y):
            filename = self.future_dataset
            with open(filename, 'r') as f:
                rows = list(csv.reader(f))
                cell = rows[y][x]
                return cell

        preds_dict = preds.to_dict()
        preds_list = [x for key, x in preds_dict['Output Values'].items()]
        changes = []
        for x, y in enumerate(preds_list):
            yester = read_cell(x+6, -1)
            yester = float(yester)
            change = y - yester
            changes.append(change)
        
        formatted_changes = []
        for change in changes:
            change = round(float(change), ndigits=3)
            formatted_changes.append(change)
        
        percent_changes = formatted_changes
        headlines = pd.read_csv(self.news_data).title.tolist()

        scores = classify(headlines)

        final_score = round(float(sum(scores)), ndigits=3)

        weight = final_score + 1

        preds_dict = preds.to_dict()
        outputs = preds_dict['Output Values']
        weighted_outputs = []
        for output in outputs:
            num = outputs[output]
            weighted_output = num * weight
            weighted_outputs.append(weighted_output)

        formatted_weighted_outputs = []
        for x in weighted_outputs:
            formatted = round(float(x), ndigits=2)
            formatted_weighted_outputs.append(formatted)
        
        weighted_outputs_d = {"Prediction Labels":self.labels, "Weighted Outputs":formatted_weighted_outputs}
        weighted_outputs_df = pd.DataFrame.from_dict(weighted_outputs_d)

        return weighted_outputs_df

    def print_past_date(self, date):
        """
        This function prints the Open, High, Low, Close, and Adj Close of any date in the past.
        Note that the date must be formatted as YYYY-MM-DD
        """

        all_data = pd.read_csv(self.dataset)
        dates = all_data['RealDate']
        dates = dates.tolist()
        idx = dates.index(date)

        prices = all_data.loc[all_data['Date'] == idx + 1].T
        prices.columns = ['']
        prices.index.name = None
        what_to_drop = []
        what_to_drop.extend(self.test_attrs)
        prices = prices.drop(what_to_drop, axis=0)

        return prices

    def plot_2D(self, x, y):
        """
        This plots a 2D graph of an x and a y, from a data object generated by the 'fit' method.
        Note that both x and y must be either attributes or features of the data object.
        """
        x_True = x in self.attributes
        y_True = y in self.attributes
        if not all([x_True, y_True]):
            raise Exception('x and y must be features or targets inside the data object.')
        style.use("ggplot")
        plt.scatter(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_3D(self, x, y, z):
        """
        This plots a 3D graph of an x, y, and z, from a data object generated by the 'fit' method.
        """
        x_True = x in self.attributes
        y_True = y in self.attributes
        z_True = z in self.attributes
        if not all([x_True, y_True, z_True]):
            raise Exception('x, y, and z must be features or targets inside the data object.')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.data[x], self.data[y], self.data[z])
        print(type(self.data[x]))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.show()
