"""
Fits various simple models to predict stock prices.
Rather than predicting the price directly, dynamic programming is used to find the optimal times
to buy/sell stocks for a given spread between buy and sell price, then an ML model predicts the optimal action
from the simple action space (hold 0 shares, hold 1 share) each day.

The performance is surprisingly good for the USD/EUR ForEx example below,
possibly because the price generally reverts to the mean throughout the train and test data.
"""

from datetime import date

import numpy as np # 1.26
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import torch # 2.6
import torch.nn as nn
import xgboost as xgb # 3.0.5
import yfinance as yf # 0.2.66
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'

STOCK = 'USDEUR=X' # USD-EUR ForEx - define 1 "share" as $1 worth of EUR
SPREAD = 0.0001 # sell price is approximately 0.01% lower than buy price for USD-EUR 


class MLPClassifier(nn.Module):

    width = 256
    actions = 2

    def __init__(self, features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, self.width),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.width, self.actions),
        )

    def forward(self, x):
        # (B, F) -> (B, A)
        return self.net(x)

    def predict(self, x: np.ndarray) -> int:
        return int(torch.argmax(self(torch.as_tensor(x, dtype=torch.float32))).item())


class XGBWrapper(xgb.XGBClassifier):

    def predict(self, x, **kwargs):
        return super().predict(x[np.newaxis,:], **kwargs)[0]


def get_optimal_actions(prices: np.ndarray) -> np.ndarray:
    """
    Use dynamic programming to step backwards through time and determine the optimal time to buy and sell.
    
    If the spread was zero, the optimal would be to buy in all valleys and sell at all peaks,
    but the finite spread means we must avoid acting too often.
    """
    dp = {}
    hist = {}

    def max_profit(day, own):
        """
        Recursively determine maximimum profit from `day` to end of `prices` given whether or not we `own` the stock.
        """
        if (day, own) in dp:
            return dp[(day, own)]

        else:
            # must sell on last day
            if day == len(prices) - 1:
                action = -own 
                val = own * (prices[day] * (1 - SPREAD))

            else:
                # value of doing nothing
                val = max_profit(day + 1, own)
                action = 0

                # value of buying
                if own:
                    vs = prices[day] * (1 - SPREAD) + max_profit(day + 1, 0)
                    if vs > val:
                        val = vs
                        action = -1
                
                # value of selling
                else:
                    vb = -prices[day] + max_profit(day + 1, 1)
                    if vb > val:
                        action = 1
                        val = vb
            
            # record max value and corresponding action
            dp[(day, own)] = val
            hist[(day, own)] = action
            return val
    
    # optimise
    max_profit(0, 0)
    
    # step back to retrieve solution
    actions = []
    pos = (0, 0)
    while pos[0] < len(prices):
        actions += [hist[pos]]
        pos = (pos[0] + 1, pos[1] + hist[pos])
        
    return np.cumsum(actions)


def download_data(ticker: str, start: date, end: date = date.today()) -> pd.DataFrame:
    """Download historical stock price data using yfinance."""
    print(f"Loading data for {ticker}")
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)
    df.columns = [col[0] for col in df.columns]
    df.drop("Volume", axis=1, inplace=True)
    return df


def handle_nans(df: pd.DataFrame) -> None:
    """Set NaNs in the data to zero."""
    df.fillna(method="bfill", inplace=True)  # backwards fill nan values
    df.fillna(0, inplace=True)  # fill remaining nan values with 0
    assert df.isnull().sum().sum() == 0


def add_features(df: pd.DataFrame) -> None:
    """Add classical trading signals as features (to be used by non-temporal models)."""
    df["SMA_3"] = df["Close"].rolling(window=3).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["pct_1"] = df['Close'].diff() / df['Close']
    df["pct_3"] = df['SMA_3'].diff() / df['SMA_3']
    df["pct_10"] = df['SMA_10'].diff() / df['SMA_10']
    df["SMA_3_10_cross"] = np.where(df["SMA_3"] > df["SMA_10"], 1, 0)
    df['SMA_3_10_signal'] = df["SMA_3_10_cross"].diff()
    df["sin_day"] = np.sin(df.index.dayofweek * 2 * np.pi / 7)
    df["cos_day"] = np.cos(df.index.dayofweek * 2 * np.pi / 7)
    df.dropna(axis=0, how="any", inplace=True)


def backtest(features: np.ndarray, prices: np.ndarray, model: MLPClassifier):
    """Simulate the profit made using the agent."""
    features[:, -1] = np.nan # remove previous position feature before backtesting

    # start with $1 cash and zero shares
    pos = 0
    cash = 1
    pos_hist = []
    cash_hist = []

    for i in range(len(prices)):
        # update previous position feature
        features[i,-1] = pos

        # agent predicts whether we should own or not
        pred_pos = model.predict(features[i,:])

        # sell
        if pos == 1 and ((i == len(prices) - 1) or (pred_pos == 0)):
            cash += prices[i] * pos * (1 - SPREAD)
            pos = 0

        # buy
        elif pos == 0 and pred_pos == 1:
            pos = 1
            cash -= prices[i]

        # record        
        cash_hist += [cash]
        pos_hist += [pos]

    return np.asarray(cash_hist), np.asarray(pos_hist) * prices


def train_mlp(x_train, y_train, x_test, y_test) -> MLPClassifier:
    """Train an MLP model."""
    print("----- MLP -----")
    # hyperparameters
    epochs = 500
    batch_size = None
    lr = 1e-3
    steps_per_epoch = 1 if batch_size is None else int(x_train.shape[0] / batch_size)
    
    # initialise
    model = MLPClassifier(x_train.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.3, patience=100, min_lr=1e-3)

    for e in range(epochs + 1):

        train_losses = []
        model.train()

        for _ in range(steps_per_epoch):
            # get batch
            if batch_size is None:
                x_batch = x_train
                y_batch = y_train
            else:
                idx_batch = torch.randint(0, x_train.shape[0], size=(batch_size,))
                x_batch = x_train[idx_batch, :]
                y_batch = y_train[idx_batch]

            # train
            optimiser.zero_grad()
            logits = model(x_batch)
            loss = nn.functional.cross_entropy(logits, y_batch).mean()
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        # test
        model.eval()
        test_logits = model(x_test)
        test_loss = nn.functional.cross_entropy(test_logits, y_test).mean().item()
        if e % 50 == 0:
            print(f"Epoch {e:03d} | LR {scheduler.get_last_lr()[0]:.1e} | Train Loss {np.mean(train_losses):.3f} | Test Loss {test_loss:.3f}")
        scheduler.step(test_loss)

    return model


def train_xgb(x_train, y_train, x_test, y_test):
    """Train Extreme Gradient Boosted Trees model."""
    print("----- XGBoost -----")
    model = XGBWrapper(tree_method="exact", early_stopping_rounds=10)
    model.fit(x_train.numpy(), y_train.numpy(), eval_set=[(x_test.numpy(), y_test.numpy())])
    return model


def train_and_test_all_models():
    """Train and backtest various models."""

    df = download_data(ticker=STOCK, start = date(2022, 9, 1))
    # add_features(df)
    optimal_actions = get_optimal_actions(df["Close"].to_numpy())
    prev_actions = np.insert(optimal_actions[:-1], 0, 0)
    df["prev_position"] = prev_actions

    # training/test split - no validation set for this simple example
    split_date = date(2024, 9, 1)
    is_train = df.index.date < split_date
    x_train = torch.as_tensor(df[is_train].to_numpy(), dtype=torch.float32)
    y_train = torch.as_tensor(optimal_actions[is_train], dtype=torch.int64)
    x_mu, x_sigma = x_train.mean(dim=0), x_train.std(dim=0)
    x_mu[-1], x_sigma[-1] = 0, 1 # don't normalise previous action
    x_train = (x_train - x_mu) / x_sigma
    x_test = torch.as_tensor(df[~is_train].to_numpy(), dtype=torch.float32)
    x_test = (x_test - x_mu) / x_sigma
    y_test = torch.as_tensor(optimal_actions[~is_train], dtype=torch.int64)
    
    # baseline
    print("---- Baseline -----")
    print("always holding zero shares")
    logits = torch.zeros((y_test.shape[0], 2), dtype=torch.float32)
    logits[:,0] = 1.0
    loss = nn.functional.cross_entropy(logits, y_test).mean().item()
    print(f"Test loss: {loss:.3f}")

    # train all the models
    mlp_model = train_mlp(x_train, y_train, x_test, y_test)
    xgb_model = train_xgb(x_train, y_train, x_test, y_test)

    # simulate using the trained agent on the test data
    prices_test = df[~is_train]["Close"].to_numpy()
    cash_mlp, position_mlp = backtest(x_test, prices_test, mlp_model)
    cash_xgb, position_xgb = backtest(x_test, prices_test, xgb_model)

    # plot all models
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Asset Price", "Portfolio Value", "Shares Held"))
    f.add_trace(
        go.Scatter(y=prices_test, mode="lines", name="Stock Price", line_color="black", showlegend=False),
        row=1, col=1,
    )
    f.add_trace(
        go.Scatter(y=cash_mlp + position_mlp, mode="lines", name="MLP", legendgroup="MLP", line_color="blue"),
        row=2, col=1,
    )
    f.add_trace(
        go.Scatter(y=cash_xgb + position_xgb, mode="lines", name="XGB", legendgroup="XGB", line_color="red"),
        row=2, col=1,
    )
    f.add_trace(
        go.Scatter(
            y=np.asarray(position_mlp > 0, dtype=np.float32),
            mode="lines", name="MLP", legendgroup="MLP", showlegend=False, line_color="blue",
        ),
        row=3, col=1,
    )
    f.add_trace(
        go.Scatter(
            y=np.asarray(position_xgb > 0, dtype=np.float32),
            mode="lines", name="XGB", legendgroup="XGB", showlegend=False, line_color="red",
        ),
        row=3, col=1,
    )
    f.show()


if __name__ == "__main__":
    train_and_test_all_models()