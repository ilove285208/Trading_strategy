import csv
import numpy as np
import os
import torch


# 讀取數據
class LoadStockData:
    def __init__(self, stock_name, fields, filename, time_period):
        self.stock_name = [stock_name]
        self.fields = fields
        
        # set the number of stock and the maximal number of stocks
        self.no_stock = 1  # number of stock
        self.max_no_stock = 10  # maximal number of stocks

        field, date, all_price = LoadStockData.LoadCSVData(filename)

        # find the index of the start_date and end_date
        start_index = np.argwhere(date >= time_period[0])[0, 0]
        end_index = np.argwhere(date <= time_period[-1])[-1, 0]
        # print(end_index,start_index)

        # clip the date and prices intervals
        self.dates = date[start_index : end_index + 1]
        self.all_price = all_price[start_index : end_index + 1]

        # collect the prices
        self.prices = np.zeros((len(self.dates), len(self.fields), self.max_no_stock), np.float32)
        # get the prices we need with field
        
        for idx, field in enumerate(self.fields):
            field_idx = self.fields.index(field)
            self.prices[:, idx, 0] = all_price[start_index : end_index + 1, field_idx]
            

    def AddNewData(self, stock_names, filenames):
        # make sure stock names and data_files have the same length
        assert len(stock_names) == len(
            filenames
        ), "StockMarket.add_stocks(): lenghts of names and files must be the same"

        for name, file in zip(stock_names, filenames):
            print(f"> adding stock market: {name} ")

            # load the data from file by invoking the LoadCSVData() to load the stock data
            fields, dates, prices = LoadStockData.LoadCSVData(file)

            # find the index of the start_date and end_date
            start_index = np.argwhere(dates >= self.dates[0])[0, 0]
            end_index = np.argwhere(dates <= self.dates[-1])[-1, 0]

            # clip the date and prices intervals
            dates = dates[start_index : end_index + 1]
            prices = prices[start_index : end_index + 1]

            # get the field indices in fields
            indices = [fields.index(field) for field in self.fields if field in fields]

            if len(indices) != len(self.fields):
                raise Exception("StockMarket.append(): missing fields")

            # select the data in the desired fields
            selected_prices = np.zeros((len(dates), len(indices)))

            for idx in range(len(indices)):
                selected_prices[:, idx] = prices[:, indices[idx]]

            # append the selected prices to self.prices
            for idx in range(len(self.dates)):
                # find the appended_idx that dates[appended_idx] <= self.dates[idx]
                appended_idx = np.argwhere(dates <= self.dates[idx])[-1, 0]
                # print(appended_idx)
                # append the data
                self.prices[idx, :, self.no_stock] = selected_prices[appended_idx]

            # append the stock names
            self.stock_name.append(name)

            # increase the number of stock
            self.no_stock = self.no_stock + 1

    def get_prices(self, date_interval, field_names, stock_names):
        if date_interval is None:
            date_interval = (self.dates[0], self.dates[-1])

        # check start_date and end_date
        assert (
            date_interval[0] <= date_interval[1]
        ), "start_date must be less than end_date"

        # find the index of the start_date and end_date
        start_idx = np.argwhere(self.dates >= date_interval[0])[0, 0]
        end_idx = np.argwhere(self.dates <= date_interval[1])[-1, 0]

        dates = self.dates[start_idx : end_idx + 1]

        # get the field indices in self.fields and self.stock_names
        field_indices = [
            self.fields.index(field) for field in field_names if field in self.fields
        ]
        stock_indices = [
            self.stock_name.index(name)
            for name in stock_names
            if name in self.stock_name
        ]

        # indexing by dates
        prices = self.prices[start_idx : end_idx + 1]

        # indexing by fields
        prices = prices[:, np.array(field_indices), :]

        # indexing by stock names
        prices = prices[:, :, np.array(stock_indices)]

        return dates, prices

    @classmethod
    def LoadCSVData(cls, filename):
        # 讀取數據
        data_path = "./" + filename

        date = []
        all_price = []
        with open(data_path, "r") as file:  # 讀取csv檔案內容
            Read_Data = csv.reader(file)
            title = next(Read_Data)
            title = title[1:]
            for line in Read_Data:
                date.append(line[0])
                all_price.append([float(value) for value in line[1:]])

            date = [float(value) for value in date]
            date = np.array(date)
            all_price = np.array(all_price)
            
        return title, date, all_price 


# main program
if __name__ == "__main__":
    # load data
    aux_market_names = [
        "S&P_500",
        "Dow_Jones",
        "NASDAQ",
        "PHLX",
        "N225",
        "FTSE",
        "FCHI",
        "DAX",
        "000001.SS",
    ]
    filename = "StockData/TWII.csv"
    data_period = (19990101, 20201231)
    Stock_Data = LoadStockData("TWII", ["close","change"], filename, data_period)
    Stock_Data.AddNewData(
        aux_market_names,
        [
            "StockData/S&P_500.csv",
            "StockData/Dow_Jones.csv",
            "StockData/NASDAQ.csv",
            "StockData/PHLX.csv",
            "StockData/N225.csv",
            "StockData/FTSE.csv",
            "StockData/FCHI.csv",
            "StockData/DAX.csv",
            "StockData/000001.SS.csv",
        ],
    )

    # get the prices of all stock markets
    selected_market_names = ["TWII"] + aux_market_names
    # selected_market_names = ["TWII"] 
    dates, prices = Stock_Data.get_prices(data_period, ["change"], selected_market_names)
