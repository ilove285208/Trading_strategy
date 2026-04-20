# TCNDQN

這是一個以台股加權指數 (`TWII`) 為主要交易標的的研究型專案，使用 `Temporal Convolutional Network (TCN)` 擷取時間序列特徵，再搭配 `Deep Q-Network (DQN)` 學習每日交易動作。

專案核心目標是讓 agent 根據台股與多個國際指數的歷史變化，輸出三種動作之一：

- `0 = sell`
- `1 = hold`
- `2 = buy`

Current implementation uses target-position semantics with `desired_position = action - 1`:
- `0 -> short (-1)`
- `1 -> flat (0)`
- `2 -> long (+1)`

專案同時保留兩條實驗線：

- `TCNDQN`：使用市場特徵 + 持倉成本/未實現損益特徵
- `TCNDQN_noC`：只使用市場特徵，不使用成本狀態

## 專案在做什麼

整體流程可分成四段：

1. 抓取並整理指數資料
2. 將多市場時間序列對齊成模型輸入
3. 用 TCN + DQN 訓練交易 agent
4. 針對不同季度或不同趨勢區間做回測與比較

目前主流程的 reward 已改成較接近實盤的版本：以每日持倉損益為主，並在換倉/平倉時扣除手續費與滑價。

## 模型架構

### 1. TCN 特徵抽取

[`TCN_Block.py`](./TCN_Block.py) 使用多層 dilation residual block 處理時間序列，從固定視窗長度內的歷史變化中萃取特徵。

### 2. DQN 動作價值

[`DQN_Block.py`](./DQN_Block.py) 使用全連接網路，根據 TCN 輸出的高階特徵計算三個動作的 Q-value。

### 3. TCN + DQN 整合

[`TCNDQN.py`](./TCNDQN.py) 會分開處理兩種輸入：

- 市場變化序列
- 資產狀態序列（平均成本、未實現損益比率）

兩組特徵在最後串接，再交給 DQN 決定動作。

無成本版本則由 [`TCNDQN_noC.py`](./TCNDQN_noC.py) 處理，只使用市場特徵。

## 資料來源與前處理

[`get_stock_data.ipynb`](./get_stock_data.ipynb) 透過 `yfinance` 下載資料，輸出到 `StockData/`。

目前專案使用的主要市場包含：

- `TWII`
- `S&P_500`
- `Dow_Jones`
- `NASDAQ`
- `PHLX`
- `N225`
- `FTSE`
- `FCHI`
- `DAX`
- `000001.SS`

[`LoadStockData.py`](./LoadStockData.py) 負責：

- 讀取 CSV
- 依日期區間裁切資料
- 對齊多市場時間軸
- 依指定欄位輸出模型需要的 tensor 形狀

常用欄位是：

- `close`
- `change`

其中 `change` 來自收盤價日報酬率百分比。

## 環境與 reward 設計

[`TCNDQN_Environment.py`](./TCNDQN_Environment.py) 是有成本狀態的環境。

它會維護：

- 當前持倉方向與張數
- 平均成本
- 未實現損益
- 已實現損益
- 成本特徵序列

### 狀態

模型每一步主要看到兩種輸入：

- `price state`：各市場在最近 `tcn_window` 內的 `change` 序列
- `cost state`：平均成本與未實現損益比率的序列

### 動作

- `sell`：目標部位設為空頭
- `hold`：目標部位設為空手
- `buy`：目標部位設為多頭

目前主環境使用的是「target position」語義，也就是每一步直接指定目標部位為空頭、空手或多頭。

### 訓練 reward

訓練時 reward 使用下一交易日的持倉報酬，並扣除當步驟發生的交易成本：

- `reward = position * daily_return - fee - slippage`
- episode 結束時若仍持倉，會自動平倉並再扣一次成本
- `reward_scale` 用來放大數值尺度，方便穩定訓練

### 測試 reward

回測時沿用相同的 daily PnL reward 定義，再另外統計累積報酬、Sharpe、CAGR、最大回撤與 turnover。

## 訓練流程

目前可用 [`train.py`](./train.py) 或 [`tcndqn_train.ipynb`](./tcndqn_train.ipynb) 執行訓練；正式重複實驗建議優先使用 `train.py`。

流程如下：

1. 設定季度或訓練區間
2. 載入 `TWII` 與輔助市場資料
3. 建立 `Environment`
4. 建立 `DQNAgent`
5. 跑多個 episode
6. 儲存每個 episode 的模型權重與圖表

目前訓練使用的 DQN 元件包含：

- replay buffer
- target network
- Double DQN target
- Dueling DQN head
- epsilon-greedy
- `SmoothL1Loss`
- gradient clipping

無成本版本使用 [`tcndqn_train_noC.ipynb`](./tcndqn_train_noC.ipynb)。

## 回測流程

目前可用 [`backtest.py`](./backtest.py) 或 [`tcndqn_test.ipynb`](./tcndqn_test.ipynb) 進行回測；正式批次比較建議優先使用 `backtest.py`。

它會：

1. 指定季度或趨勢區間
2. 載入該區間所有訓練好的 `model*.pt`
3. 對每個 checkpoint 進行回測
4. 記錄每個模型的總 reward
5. 輸出動作序列到 `strategy/`
6. 繪製 reward 比較圖

無成本版本使用 [`tcndqn_test_noC.ipynb`](./tcndqn_test_noC.ipynb)。

## 資料夾說明

### 原始碼

- [`TCNDQN.py`](./TCNDQN.py)：有成本狀態的主模型
- [`TCNDQN_noC.py`](./TCNDQN_noC.py)：無成本狀態版本
- [`TCNDQN_DQNAgent.py`](./TCNDQN_DQNAgent.py)：訓練用 agent
- [`TCNDQN_DQNAgent_noC.py`](./TCNDQN_DQNAgent_noC.py)：無成本版本 agent
- [`TCNDQN_Environment.py`](./TCNDQN_Environment.py)：訓練/回測環境
- [`TCNDQN_Environment_noC.py`](./TCNDQN_Environment_noC.py)：無成本版本環境
- [`LoadStockData.py`](./LoadStockData.py)：CSV 載入與多市場對齊
- [`TCN_Block.py`](./TCN_Block.py)：TCN block
- [`DQN_Block.py`](./DQN_Block.py)：DQN head
- [`train.py`](./train.py)：正式訓練腳本
- [`validate.py`](./validate.py)：驗證區間自動選模腳本
- [`backtest.py`](./backtest.py)：正式回測腳本
- [`signal.py`](./signal.py)：每日 flat/long/short 訊號入口
- [`paper_trade.py`](./paper_trade.py)：每日模擬持倉與資產台帳
- [`live.py`](./live.py)：實盤下單骨架，支援 dry-run / file / custom adapter
- [`tcndqn_cli_utils.py`](./tcndqn_cli_utils.py)：訓練/回測共用工具

### Notebook

- [`get_stock_data.ipynb`](./get_stock_data.ipynb)：抓資料並輸出 CSV
- [`tcndqn_train.ipynb`](./tcndqn_train.ipynb)：主訓練流程
- [`tcndqn_test.ipynb`](./tcndqn_test.ipynb)：主回測流程
- [`tcndqn_train_noC.ipynb`](./tcndqn_train_noC.ipynb)：無成本訓練
- [`tcndqn_test_noC.ipynb`](./tcndqn_test_noC.ipynb)：無成本回測
- [`test_correlation.ipynb`](./test_correlation.ipynb)：指數相關性分析
- [`test_plot.ipynb`](./test_plot.ipynb)：損失函數視覺化比較

### 輸入資料

- `StockData/`：所有市場 CSV

### 訓練輸出

- `model/<run_name>/`：模型權重、Q-value 圖、Action distribution 圖、最終成本狀態、config 與 training summary

### 驗證輸出

- `model/<run_name>/validation/summary.csv`：驗證區間各 checkpoint 指標
- `model/<run_name>/validation/ranking_by_<metric>.csv`：依選模指標排序的排名
- `model/<run_name>/validation/selected_checkpoint.json`：自動挑出的 checkpoint
- `model/<run_name>/selected_checkpoint.json`：最新選模結果的便利用途副本

### 回測輸出

- `model/<run_name>/backtest/summary.csv`：各 checkpoint 指標
- `model/<run_name>/backtest/best_checkpoint.json`：最佳 checkpoint 詳細結果
- `model/<run_name>/backtest/strategies/`：各 checkpoint 的動作與部位序列
- `model/<run_name>/backtest/*.png`：checkpoint reward 與 equity curve 圖表

### 訊號輸出

- `model/<run_name>/signals/YYYYMMDD_signal.json`：指定日期的訊號、Q-value 與最近回放動作

### Paper Trading 輸出

- `model/<run_name>/paper_trading/state.json`：模擬帳戶最新狀態
- `model/<run_name>/paper_trading/ledger.csv`：每日模擬交易台帳
- `model/<run_name>/paper_trading/runs/YYYYMMDD_paper_trade.json`：單日執行明細

### Live Trading 輸出

- `model/<run_name>/live/YYYYMMDD_live.json`：實盤訊號與下單決策紀錄
- `broker_outbox/` 或自訂 outbox：`file` broker 模式產出的訂單指令 JSON

## 歷史輸出格式

目前工作區中已經保留大量歷史實驗輸出，例如：

- `model/16season-1/`、`model/16season-2/`、`model/16season-3/`
- `model/up_trend/`
- `model/down_trend/`
- `model/flat_trend/`
- `model/all_trend/`
- `model/2025_trend/`
- `model/No_Cost/`

其中 `16season-*` 對應的是以季度切分的訓練與回測實驗。

## 依賴套件

這個專案沒有提供 `requirements.txt`，但從程式內容可推得至少需要：

- `python`
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `yfinance`
- `python-dateutil`
- `tqdm`
- `seaborn`
- `jupyter`

[`check_version.py`](./check_version.py) 目前是用來檢查本機 `torch`、`torchaudio`、`torchvision` 與 CUDA/CuDNN 版本。

## 建議使用順序

若要重新跑整份專案，建議順序如下：

1. 先執行 [`get_stock_data.ipynb`](./get_stock_data.ipynb) 更新 `StockData/`
2. 執行 [`train.py`](./train.py) 產生模型
3. 執行 [`validate.py`](./validate.py) 在 validation 區間挑 checkpoint
4. 執行 [`backtest.py`](./backtest.py) 在 test 區間做最終評估
5. 執行 [`signal.py`](./signal.py) 產生日常訊號
6. 執行 [`paper_trade.py`](./paper_trade.py) 做每日模擬持倉
7. 需要串接券商時，執行 [`live.py`](./live.py) 做 preview 或送單
8. 若要做基線比較，再跑 `*_noC.ipynb`

## 目前限制

- `noC` 基線版本仍主要依賴舊 notebook / 舊環境邏輯，尚未完全同步到新版 daily PnL reward。
- 缺少 `requirements.txt` 或環境鎖定檔，重現性有限。
- reward 目前是單位化報酬率版本，尚未納入資金曲線、部位 sizing、保證金與合約乘數。
- 部分歷史輸出資料夾命名與現行程式中的預設路徑不完全一致，重跑實驗前建議先確認 notebook 內的輸出路徑設定。

## 備註

這份專案比較像「研究紀錄 + 實驗輸出倉庫」，不是已封裝完成的產品型專案。若下一步要繼續維護，建議優先做三件事：

1. 補 `requirements.txt`
2. 補 train/validation/test 或 walk-forward 切分，避免只看單段結果
3. 把 reward 再推進到資金曲線與風險約束版本
