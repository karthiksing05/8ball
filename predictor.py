__doc__ = """
Description of this price predictor:
This is my Investor's 8ball. It focuses on using news headlines, previous stock prices, and machine
learning to calculate the price of an item on the "Yahoo Finance Market on any date. Currently, we are 
starting off with an sklearn framework for linear regression, which will be used to predict the stock prices, 
and then, because numbers are not enough to express what happens in the real world, we will use news headlines 
and Reddit and Twitter posts to calculate a weight, which we will multiply each of the stock predictions by to 
get our final result.


Some Important Information:
-Attributes are the input data, they can also be called features
-Labels are the output data, they can also be called targets



By: Karthik Singaravadivelan
"""

# All the imports go below
# These modules are for yahoo finance information purposes and news scraping
import requests

# My usual machine learning imports:
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# These are for visualization
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

# XGBoost regressor!
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBRegressor

# imports included with python that I need
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import datetime
import csv
import time
from io import StringIO

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

class MarketPredictor(object): # v2
    """
    My Market Predictor class (v2)

    By Karthik Singaravadivelan

    This class predicts the price of something on the "Yahoo Finance" market.
    There is no dataset needed, and there are very little commands involved.
    
    First, you must create an instance of this class with the input of a ticker.
    Then, use the load_data() method to load your dataset for the predicting.
    After you've loaded your data, use the predict method to predict your prices.

    Note: The accuracy of this model ranges from 0.8 to 0.98 based on the attributes and data given.

    You can change the amount of data the model recieves, however it is preferable to keep the 
    start and end time variables to "None" to recieve maximum data. If you end up using the variables,
    then keep the format of the dates in "YYYY-MM-DD". Note that the dates are inclusive, so if you
    want the date range to be 2012-05-01 through 2021-05-01, you will have to specify those dates.

    """
    # the constructor
    def __init__(self, ticker, dayspast=60, start_time=None, end_time=None):
        self.ticker = ticker.upper()
        self.attributes = None
        self.labels = ["Open","High","Low","Close","Adj Close"]
        self.model = None # a variable to hold the regression model.
        self.model_path = "stockmodel.pickle"
        self.dayspast = dayspast
        self.start_time = start_time
        self.end_time = end_time
        self.data = None # a variable to hold the data python object.
        self.dataset = R"data\{}.csv".format(str(self.ticker))# actual data
        self.future_dataset = R"data\Future{}.csv".format(str(self.ticker))# dataset for predictions
        self.benchmark = R"data\benchmark.csv"
        self.benchmark2 = R"data\benchmark2.csv"
        self.test_attrs = None
        self.num_test_attrs = None
        self.preds_dict = None

    def load_data(self):
        """
        This function will load the data from the Yahoo Finance website with BeautifulSoup4
        and put it in a csv file.
        """

        def read_cell(x, y):
            filename = self.benchmark
            with open(filename, 'r') as f:
                rows = list(csv.reader(f))
                cell = rows[y][x]
                return cell

        with open("error.txt", "r") as f:
            ERROR_MESSAGE = str(f.read())

        if os.path.exists(self.dataset):
            os.remove(self.dataset)

        if os.path.exists(self.future_dataset):
            os.remove(self.future_dataset)

        if os.path.exists(self.model_path):
            os.remove(self.model_path)

        if os.path.exists(self.benchmark):
            os.remove(self.benchmark)
        
        if os.path.exists(self.benchmark2):
            os.remove(self.benchmark2)

        ################################
        # Historical Stock Data
        r = None
        x = 0
        while r == None:
            url = f'https://query2.finance.yahoo.com/v7/finance/download/{self.ticker}?range=20y&interval=1d&events=history&includeAdjustedClose=true'
            # print(url)
            r = requests.get(url, allow_redirects=True)
            if str(r.text) == ERROR_MESSAGE:
                x += 1
                print(f"Received error message {x} times!")
                r = None
                time.sleep(5)
            else:
                break

        file = StringIO(r.text)
        reader = csv.reader(file)
        csvdata = list(reader)

        # turning everything into a csv:
        # this formats everthing along the way as well.
        # Basically what this loop does is, instead of just writing a list to the end of the 
        # csv file, this is turning everthing into floats and integers
        # so the prediction script can read everything properly.
        for listrow in range(len(csvdata)):
            shifter = 0
            quitloop = False
            row = csvdata[listrow]
            formatted_row = []
            current_sdate = row[0]
            if listrow > 0:
                for elem in row:
                    try:
                        if row.index(elem) != 6:
                            formatted_row.append(float(elem))
                    except ValueError:
                        if elem == "null":
                            quitloop = True
                            break
                        else:
                            formatted_row.append(listrow - shifter)
                formatted_row.append(current_sdate)
                row = formatted_row
            else:
                row.pop(-1)
                row.append("RealDate")
            if quitloop == True:
                quitloop = False
                shifter += 1
                continue
            file_to_write_to = self.benchmark
            # Finally, writing the csv file
            with open(file_to_write_to, 'a', newline="") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(row)

        benchmark_lines = []
        with open(self.benchmark, 'r', newline="") as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                benchmark_lines.append(row)

        new_benchmarks = []
        for benchmark_line in benchmark_lines:
            benchmark_line.pop(-1)
            new_benchmarks.append(benchmark_line)
        
        benchmark_lines = new_benchmarks

        header_template = benchmark_lines.pop(0)
        header_template.pop(0)
        final_header_row = ["Date"]
        for num in range(self.dayspast):
            formatted_headers = []
            formatted_headers.extend(header_template)
            if num > 0:
                for idx in range(len(formatted_headers)):
                    formatted_headers[idx] = str(formatted_headers[idx]) + str(num)
            final_header_row.extend(formatted_headers)
        final_header_row.append("RealDate")

        with open(self.benchmark2, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(final_header_row)

        self.attributes = final_header_row[0:-1]
        self.test_attrs = [x for x in self.attributes if x not in self.labels] # all the attributes for testing the model
        self.num_test_attrs = len(self.test_attrs)

        for line_num in range(len(benchmark_lines)):
            try:
                real_line_num = line_num + self.dayspast - 1
                final_line = []
                for day in range(self.dayspast):
                    my_val = real_line_num - day
                    line = benchmark_lines[my_val]
                    final_line.extend(line)
                with open(self.benchmark2, 'a', newline="") as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    for num in range(self.dayspast):
                        if num > 0:
                            idx = (num * 6) - (num - 1)
                            final_line.pop(idx)
                    cdate = read_cell(6, (real_line_num + 1))
                    final_line.append(cdate)
                    today = datetime.datetime.now().strftime("%Y-%m-%d")
                    if cdate != today:
                        writer.writerow(final_line)
            except IndexError:
                break

        if self.start_time and self.end_time:
            with open(self.dataset, 'a', newline="") as output_file:
                nwriter = csv.writer(output_file)
                with open(self.benchmark2, 'r') as input_file:
                    reader = csv.reader(input_file)

                    line_index = 0 # debugging 

                    for row in reader:
                        line_index += 1 
                        date = row[-1]
                        try:

                            startyear = int(self.start_time[0:4])
                            startmonth = int(self.start_time[5:7])
                            startday = int(self.start_time[8:10])

                            endyear = int(self.end_time[0:4])
                            endmonth = int(self.end_time[5:7])
                            endday = int(self.end_time[8:10])

                            currentyear = int(date[0:4])
                            currentmonth = int(date[5:7])
                            currentday = int(date[8:10])

                            current_obj = datetime.datetime(
                                year=currentyear, 
                                month=currentmonth, 
                                day=currentday
                            )

                            start_obj = datetime.datetime(
                                year=startyear,
                                month=startmonth,
                                day=startday
                            )

                            end_obj = datetime.datetime(
                                year=endyear,
                                month=endmonth,
                                day=endday
                            )

                            if (current_obj >= start_obj) and (current_obj <= end_obj):
                                nwriter.writerow(row)
                        except Exception as e:
                            if str(e) == "invalid literal for int() with base 10: 'Real'":
                                nwriter.writerow(row)
                            else:
                                print(e)
        
        else:
            with open(self.dataset, 'a', newline="") as output_file:
                nwriter = csv.writer(output_file)
                with open(self.benchmark2, 'r') as input_file:
                    reader = csv.reader(input_file)
                    for row in reader:
                        nwriter.writerow(row)

        os.popen(f'copy {self.dataset} {self.future_dataset}')
        # os.remove(self.benchmark)
        # os.remove(self.benchmark2)

    def fit_inital(self, filename:str=None):
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

        X = data.drop(predict, 1).values
        y = data[predict].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        append_new_line("log.txt", "\nTraining set: {} samples".format(X_train.shape))
        append_new_line("log.txt", "Test set: {} samples".format(X_test.shape))

        inside_model = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.045,
            max_depth=5,
            n_estimators=475,
            gamma=6.5e-6,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=10
        )

        model = MultiOutputRegressor(inside_model)
        model.fit(X_train, y_train)

        self.model = model

        append_new_line("log.txt", "\nFitting/Tuning Finished at "+str(datetime.datetime.now()))

    def predict(self, date:str, filename:str=None):
        """
        This predictor method predicts stocks, given a date in the form of YYYY-MM-DD.
        It will predict any date, however the farther into the future you choose, the longer it will take to predict the stock price, and the less reliable the machine will be.
        """
        

        # What this model is doing is, it's predicting one day into the future, 
        # then it adds the values it gets to the Future(ticker).csv dataset.
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

        print("Model Used For Prediction: " + str(get_model_name(self.model)))
        # These lines are to see how many times the "for" loop should run.

        if not self.end_time:
            today = datetime.date.today()
            today = (today - datetime.timedelta(days=1))
            date2 = datetime.date(today.year, today.month, today.day)
        else:
            endyear = int(self.end_time[0:4])
            endmonth = int(self.end_time[5:7])
            endday = int(self.end_time[8:10])
            date2 = datetime.date(endyear, endmonth, endday)
        date1 = datetime.date(year, month, day)
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
            

            linear = self.model

            linear.fit(X_train, y_train)

            # print(coef_str)
            # print(y_in_str)

            append_new_line("log.txt", "\nPredictions Fitting Finished at "+str(datetime.datetime.now()))

            """
            Here is where the predicting begins.
            """
            current_sd = stock_days - (days - day - 1)

            pred_values = [current_sd]
            # print(days, stock_days)
            
            for day in range(self.dayspast - 1):
                for pred_attr in self.labels:
                    cell_x = int(self.attributes.index(pred_attr))
                    cell_y = -1
                    # print(f"({cell_x}, {cell_y})")
                    pred_value = read_cell(cell_x, cell_y)
                    pred_value = float(pred_value)

                    pred_values.append(pred_value)

            my_values = np.array([pred_values])[0]
            my_values = my_values.reshape(-1, self.num_test_attrs)

            formatted_pred_values = []
            for value in my_values.tolist()[0]:
                value = round(float(value), 2)
                formatted_pred_values.append(value)

            prediction = linear.predict(my_values)
            output_values = list(prediction[0])

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

            if day == (days - 1):
                # print("\n")
                # print(df_str)
                pass

            future_data = []
            future_data.extend(pred_values)
            future_data[0] = int(future_data[0]) + 1
            for elem in reversed(list(output_values)):
                future_data.insert(1, round(elem, 5))
            # print("\nData to be plugged into Future Dataset: {}".format(future_data))

            future_data[0] = int(future_data[0])

            # print(future_data)

            last_date = stock_days - days + 1 # This variable is the last real-value date predicted by the object

            current_stock_date = date

            future_data.append(current_stock_date)

        print("\nSee log.txt for more details about the training and testing of the model. (And all the dataframes)")
        try:
            self.preds_dict = label_d
            return label_df
        except:
            print("You have predicted a date that has already been processed by the data.")
            print("To view the corresponding prices for this date, use the 'print_past-date'")
            print("method, please.")
            exit()

    def progressive_calibration(self, date:str):
        pass

    def print_past_date(self, date:str):
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

    def plot_2D(self, x:str, y:str):
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

    def plot_3D(self, x:str, y:str, z:str):
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
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.show()
