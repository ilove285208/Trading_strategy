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
        self.max_no_stock = 10  # initial capacity

        csv_fields, date, all_price = LoadStockData.LoadCSVData(filename)
        field_indices = self._get_field_indices(csv_fields)

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
        self.prices[:, :, 0] = all_price[start_index : end_index + 1, field_indices]
            

    def AddNewData(self, stock_names, filenames):
        # make sure stock names and data_files have the same length
        assert len(stock_names) == len(
            filenames
        ), "StockMarket.add_stocks(): lenghts of names and files must be the same"

        for name, file in zip(stock_names, filenames):
            print(f"> adding stock market: {name} ")
            self._ensure_capacity(self.no_stock + 1)

            # load the data from file by invoking the LoadCSVData() to load the stock data
            fields, dates, prices = LoadStockData.LoadCSVData(file)

            # get the field indices in fields
            indices = self._get_field_indices(fields)

            # select the data in the desired fields
            selected_prices = prices[:, indices]
            aligned_prices = self._align_prices_to_dates(dates, selected_prices, name)

            # append the selected prices to self.prices
            self.prices[:, :, self.no_stock] = aligned_prices

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
        field_indices = self._resolve_requested_indices(field_names, self.fields, "fields")
        stock_indices = self._resolve_requested_indices(stock_names, self.stock_name, "stocks")

        # indexing by dates
        prices = self.prices[start_idx : end_idx + 1]

        # indexing by fields
        prices = prices[:, np.array(field_indices), :]

        # indexing by stock names
        prices = prices[:, :, np.array(stock_indices)]

        return dates, prices

    def _get_field_indices(self, available_fields):
        """Resolve requested fields against the CSV header order."""
        indices = [available_fields.index(field) for field in self.fields if field in available_fields]

        if len(indices) != len(self.fields):
            missing_fields = [field for field in self.fields if field not in available_fields]
            raise Exception(f"StockMarket.append(): missing fields {missing_fields}")

        return indices

    def _resolve_requested_indices(self, requested_names, available_names, label):
        missing_names = [name for name in requested_names if name not in available_names]
        if missing_names:
            raise Exception(f"StockMarket.get_prices(): missing {label} {missing_names}")

        return [available_names.index(name) for name in requested_names]

    def _ensure_capacity(self, required_stocks):
        if required_stocks <= self.max_no_stock:
            return

        new_capacity = max(self.max_no_stock * 2, required_stocks)
        expanded = np.zeros(
            (self.prices.shape[0], self.prices.shape[1], new_capacity), np.float32
        )
        expanded[:, :, : self.max_no_stock] = self.prices
        self.prices = expanded
        self.max_no_stock = new_capacity

    def _align_prices_to_dates(self, source_dates, source_prices, market_name):
        """
        Align an auxiliary market to the primary market calendar by carrying
        forward the latest available observation that is not later than the
        target date.
        """
        valid_end_indices = np.argwhere(source_dates <= self.dates[-1])
        if len(valid_end_indices) == 0:
            raise Exception(
                f"StockMarket.append(): {market_name} has no data on or before {int(self.dates[-1])}"
            )

        clipped_end = valid_end_indices[-1, 0] + 1
        clipped_dates = source_dates[:clipped_end]
        clipped_prices = source_prices[:clipped_end]

        aligned_indices = np.searchsorted(clipped_dates, self.dates, side="right") - 1

        if np.any(aligned_indices < 0):
            raise Exception(
                f"StockMarket.append(): {market_name} starts at {int(clipped_dates[0])}, "
                f"which is later than the requested start date {int(self.dates[0])}"
            )

        return clipped_prices[aligned_indices]

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
