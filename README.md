# TCNDQN

`TCNDQN` 是一個以台股加權指數 (`TWII`) 為主要標的的研究型程式交易專案。  
這個專案使用 `Temporal Convolutional Network (TCN)` 擷取多市場時間序列特徵，並以 `Double + Dueling DQN` 學習日頻交易決策。

目前主流程已經整理成可重複執行的腳本，包含：

- 訓練：`train.py`
- 驗證選模：`validate.py`
- 回測：`backtest.py`
- 訊號輸出：`signal.py`
- 模擬交易：`paper_trade.py`
- 實盤骨架：`live.py`

## 專案定位

這個 repo 比較適合被理解成：

- 一個可研究、可重跑的強化學習交易實驗框架
- 一個從資料、訓練、驗證、回測到訊號輸出的完整流程範例
- 一個可延伸到 paper trading / live trading 的策略工程骨架

這個 repo 不應被理解成：

- 已經驗證可穩定獲利的實盤系統
- 已綁定券商、開箱即用的正式下單平台
- 投資建議

## 目前版本的核心設計

### 多市場輸入

主要市場與輔助市場目前包含：

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

資料由 [`get_stock_data.ipynb`](./get_stock_data.ipynb) 透過 `yfinance` 下載並整理成 `StockData/*.csv`。

### 模型結構

主模型位於 [`TCNDQN.py`](./TCNDQN.py)，由兩個 TCN 分支組成：

- `price branch`：處理各市場的 `change` 序列
- `cost branch`：處理部位相關狀態，例如平均成本與未實現損益

兩個分支的特徵最後會串接，送入 [`DQN_Block.py`](./DQN_Block.py) 中的 `Dueling DQN` head，輸出三個 action 的 Q-value。

### Action 語義

目前主環境採用 **target-position semantics**：

- `0 = sell -> short (-1)`
- `1 = hold -> flat (0)`
- `2 = buy -> long (+1)`

也就是說，模型輸出的是「目標部位」，不是單純的加碼 / 減碼指令。

### Reward 設計

目前主流程的 reward 比較接近實盤日頻邏輯：

```text
reward = position * daily_return - fee - slippage
```

特性如下：

- reward 使用下一交易日的持倉報酬
- 換倉時會扣手續費與滑價
- episode 結束時若仍持倉，會自動平倉並再扣一次成本
- `reward_scale` 用來調整 reward 尺度，方便訓練穩定

## 專案結構

### 核心模型與環境

- [`TCNDQN.py`](./TCNDQN.py)：主模型，含市場分支與成本分支
- [`TCNDQN_Environment.py`](./TCNDQN_Environment.py)：主環境，實作 target position 與 daily PnL reward
- [`TCNDQN_DQNAgent.py`](./TCNDQN_DQNAgent.py)：主訓練 agent
- [`TCN_Block.py`](./TCN_Block.py)：TCN block
- [`DQN_Block.py`](./DQN_Block.py)：Dueling DQN head

### 資料處理

- [`LoadStockData.py`](./LoadStockData.py)：CSV 載入、多市場日期對齊、欄位整理
- [`get_stock_data.ipynb`](./get_stock_data.ipynb)：更新 `StockData/`

### 主要腳本

- [`train.py`](./train.py)：訓練模型並輸出 checkpoint
- [`validate.py`](./validate.py)：在 validation 區間自動選 checkpoint
- [`backtest.py`](./backtest.py)：對 checkpoint 做回測與績效彙整
- [`signal.py`](./signal.py)：產生每日 `short / flat / long` 訊號
- [`paper_trade.py`](./paper_trade.py)：維護模擬帳戶與交易台帳
- [`live.py`](./live.py)：實盤骨架，預設支援 `dry-run` / `file` broker
- [`live_broker.py`](./live_broker.py)：broker adapter 介面與內建實作

### 研究 / 舊版內容

- [`tcndqn_train.ipynb`](./tcndqn_train.ipynb)
- [`tcndqn_test.ipynb`](./tcndqn_test.ipynb)
- [`tcndqn_train_noC.ipynb`](./tcndqn_train_noC.ipynb)
- [`tcndqn_test_noC.ipynb`](./tcndqn_test_noC.ipynb)
- [`TCNDQN_noC.py`](./TCNDQN_noC.py)
- [`TCNDQN_Environment_noC.py`](./TCNDQN_Environment_noC.py)
- [`TCNDQN_DQNAgent_noC.py`](./TCNDQN_DQNAgent_noC.py)

其中 `noC` 基線版本目前仍保留較舊的研究邏輯，**不完全等同** 主流程的 reward / validation / signal 設計。

## 快速開始

### 1. 安裝環境

本專案目前沒有提供 `requirements.txt`，建議至少準備：

- `python`
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `yfinance`
- `python-dateutil`
- `tqdm`
- `seaborn`
- `pytest`（測試時需要）

建議先依照 [PyTorch 官方安裝說明](https://pytorch.org/get-started/locally/) 安裝對應版本的 `torch`，再安裝其餘套件：

```bash
python -m pip install numpy matplotlib pandas yfinance python-dateutil tqdm seaborn pytest
```

### 2. 更新資料

執行 [`get_stock_data.ipynb`](./get_stock_data.ipynb) 重新下載 / 更新 `StockData/`。

### 3. 訓練模型

```bash
python train.py \
  --train-start-date 20170101 \
  --train-end-date 20211231 \
  --episodes 100 \
  --label twii_v1
```

輸出會寫到：

```text
model/twii_v1_20170101_20211231/
```

其中包含：

- `model*.pt`
- `config.json`
- `training_summary.json`
- `training_curves.png`
- `action_distribution.png`
- `q_value_trends.png`
- `final_cost.pt`

### 4. 驗證選模

```bash
python validate.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --validation-start-date 20220101 \
  --validation-end-date 20221231 \
  --selection-metric sharpe
```

輸出包含：

- `validation/summary.csv`
- `validation/ranking_by_sharpe.csv`
- `validation/selected_checkpoint.json`
- `selected_checkpoint.json`

### 5. 最終回測

```bash
python backtest.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --test-start-date 20230101 \
  --test-end-date 20241231
```

回測輸出包含：

- `backtest/summary.csv`
- `backtest/best_checkpoint.json`
- `backtest/*.png`
- `backtest/strategies/*.txt`

### 6. 產生每日訊號

```bash
python signal.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --signal-date 20250418
```

輸出：

- `signals/YYYYMMDD_signal.json`

### 7. 模擬交易

```bash
python paper_trade.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --signal-date 20250418 \
  --initial-cash 1000000 \
  --units-per-position 1 \
  --contract-multiplier 1
```

輸出：

- `paper_trading/state.json`
- `paper_trading/ledger.csv`
- `paper_trading/runs/YYYYMMDD_paper_trade.json`

### 8. 實盤預覽 / 寫單骨架

`live.py` 的角色是把模型訊號轉成 **target-position order**。  
它會先讀取模型在指定日期的 `short / flat / long` 訊號，再和目前部位相比，計算出應該送出的 `target_quantity` 與 `delta_quantity`，最後交給 broker adapter 處理。

這個流程適合被理解成一個 execution layer skeleton：

- 負責把研究模型輸出轉成可執行訂單
- 支援只預覽、不送單的檢查流程
- 支援把下單責任交給外部 broker bridge / adapter

每次執行都會額外輸出一份 live log：

- `model/<run>/live/YYYYMMDD_live.json`

裡面會保留本次使用的訊號、目前部位、目標部位、下單方向與 adapter 回傳結果，方便之後追蹤。

#### Dry-run 預覽

`dry-run` 適合用在正式串接券商之前，先驗證「模型訊號是否被正確翻譯成部位變化與訂單方向」。

```bash
python live.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --signal-date 20250418 \
  --symbol TXF \
  --broker dry-run \
  --current-quantity 0
```

重點如下：

- `--current-quantity` 代表你手動告訴系統目前持倉數量
- 系統會依照 `signal target` 與 `current_quantity` 計算 `delta_quantity`
- 若沒有加上 `--execute`，只會做預覽，狀態會是 `preview_only`
- 即使有加上 `--execute`，`dry-run` 也只會模擬送單，不會連任何外部券商
- 適合用來檢查 `buy / sell / hold`、目標部位換算、`order_type` 與 `time_in_force` 是否符合預期

這個模式的限制也很明確：

- 持倉是透過 CLI 參數手動提供，不是從外部系統自動同步
- 模擬結果只存在本次執行流程與輸出 log，不會幫你維護真實券商狀態
- 比較適合部署前驗證、排查訊號轉單邏輯，或做人工覆核

#### File broker 模式

`file` broker 適合已經有自己的下單橋接程式、排程器或券商整合層，但不希望讓研究框架直接持有券商連線邏輯的情境。

```bash
python live.py \
  --model-dir model/twii_v1_20170101_20211231 \
  --signal-date 20250418 \
  --symbol TXF \
  --broker file \
  --order-outbox broker_outbox \
  --execute
```

這個模式不會直接連券商，而是把訂單指令寫成 JSON，方便外部 bridge / adapter 接手。

運作方式如下：

- 若未加 `--execute`，一樣只會預覽，不會寫出訂單檔
- 加上 `--execute` 後，系統會把訂單指令寫到 `--order-outbox` 指定資料夾
- 輸出檔名格式為 `<SYMBOL>-<UTC timestamp>.json`
- adapter 狀態通常會是 `queued`，代表「已產生待外部處理的訂單指令」

輸出的 JSON 會包含：

- `symbol`
- `target_quantity`
- `current_quantity`
- `delta_quantity`
- `side`
- `order_type`
- `time_in_force`
- `signal`

如果你希望 `live.py` 不是靠人工輸入目前部位，而是從外部系統讀取，可搭配 `--broker-state-path` 提供 broker 狀態檔。  
目前支援的格式例如：

```json
{
  "positions": {
    "TXF": {
      "quantity": 1
    }
  }
}
```

或更簡化地寫成：

```json
{
  "positions": {
    "TXF": 1
  }
}
```

這樣 `live.py` 就能先讀取目前部位，再決定本次應該寫出多少張數的調整單。  
對外部系統來說，這種設計的好處是研究框架只負責「產生交易意圖」，真正的券商驗證、風控、簽核、送單與回報整合都可以留在你自己的 execution stack 中。

## 測試

目前已加入針對環境 state 對齊問題的 regression test：

- [`tests/test_environment_alignment.py`](./tests/test_environment_alignment.py)

執行方式：

```bash
python -m pytest tests/test_environment_alignment.py -q
```

這支測試主要驗證：

- `price state` 和 `cost state` 的 window 長度一致
- `get_input_data(step)` 有包含當前 timestep
- `action_execution()` 後的 `cost state` 仍對齊同一個 step

## 免責聲明

本專案僅供研究、學習與工程實作參考，不構成任何投資建議。  
請勿在未充分驗證風險、交易成本、流動性與券商執行細節的情況下直接用於真實資金交易。
