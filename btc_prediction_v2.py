import os
import pandas as pd
import numpy as np
import datetime
import time
import gymnasium as gym
from dotenv import load_dotenv
from stable_baselines3 import A2C, PPO
from binance.client import Client
from sklearn.preprocessing import StandardScaler

# 載入環境變數
load_dotenv()

# 設置Binance API
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# 技術指標列表 - 可以根據需要擴展
TECH_INDICATORS = ['ma7', 'ma25', 'ma99', 'rsi14', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']

class BinanceDataFetcher:
    """獲取Binance數據的類"""
    
    def __init__(self, api_key, api_secret):
        """初始化Binance客戶端"""
        self.client = Client(api_key, api_secret)
        print("Binance客戶端初始化成功")
    
    def fetch_historical_data(self, symbol, interval, lookback_days):
        """
        獲取歷史K線數據
        
        參數:
            symbol (str): 交易對符號 (例如 'BTCUSDT')
            interval (str): K線間隔 ('30m', '1h', '1d')
            lookback_days (int): 回溯的天數
            
        返回:
            pd.DataFrame: 處理後的數據框架
        """
        # 計算開始時間 (毫秒)
        start_time = int((datetime.datetime.now() - datetime.timedelta(days=lookback_days)).timestamp() * 1000)
        
        # 獲取K線數據
        print(f"獲取 {symbol} {interval} 數據，回溯 {lookback_days} 天...")
        
        # 將Binance間隔轉換為適當的格式
        if interval == '30m':
            binance_interval = Client.KLINE_INTERVAL_30MINUTE
        elif interval == '1h':
            binance_interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '1d':
            binance_interval = Client.KLINE_INTERVAL_1DAY
        else:
            raise ValueError(f"不支持的間隔: {interval}")
        
        # 獲取K線數據
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=binance_interval,
            start_str=start_time,
            limit=1000  # Binance API限制
        )
        
        # 創建數據框架
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # 轉換數據類型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 整理數據框架
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        # 添加交易對標識和日期索引
        df['tic'] = symbol
        df['day'] = range(len(df))
        
        print(f"獲取到 {len(df)} 條數據記錄")
        return df
    
    def get_current_price(self, symbol):
        """獲取當前價格"""
        ticker = self.client.get_ticker(symbol=symbol)
        return float(ticker['lastPrice'])

class TechnicalIndicators:
    """計算技術指標的類"""
    
    @staticmethod
    def add_indicators(df):
        """
        添加技術指標到數據框架
        
        參數:
            df (pd.DataFrame): 原始數據框架
            
        返回:
            pd.DataFrame: 添加了技術指標的數據框架
        """
        # 確保數據是排序的
        df = df.sort_values('date').reset_index(drop=True)
        
        # 移動平均線
        df['ma7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma25'] = df['close'].rolling(window=25, min_periods=1).mean()
        df['ma99'] = df['close'].rolling(window=99, min_periods=1).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # 布林帶
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['stddev'] = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['ma20'] + (df['stddev'] * 2)
        df['bb_lower'] = df['ma20'] - (df['stddev'] * 2)
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, min_periods=1, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, min_periods=1, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1, adjust=False).mean()
        
        # 補充缺失值
        df = df.ffill().bfill()
        
        return df

class FinRLEnvWrapper(gym.Env):
    """
    FinRL環境的包裝器，用於處理與Stable-Baselines3的接口兼容性
    """
    def __init__(self, df, time_window='30m'):
        """
        初始化環境包裝器
        
        參數:
            df (pd.DataFrame): 股價數據框架
            time_window (str): 時間窗口 ('30m', '1h', '1d')
        """
        super(FinRLEnvWrapper, self).__init__()
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        
        # 存儲時間窗口
        self.time_window = time_window
        
        # 獲取基本配置
        self.stock_dim = len(df['tic'].unique())
        self.tech_indicators = TECH_INDICATORS
        
        # 根據測試，FinRL返回的觀察是一個元組，其中第一個元素包含多個值
        # 計算觀察空間的維度: 1(總資產) + 1(收盤價) + 1(持倉) + len(技術指標)
        self.observation_shape = 3 + len(self.tech_indicators)
        
        # 創建原始環境
        self.env = StockTradingEnv(
            df=df,
            stock_dim=self.stock_dim,
            hmax=100,
            initial_amount=10000,  # 初始資金
            num_stock_shares=[0] * self.stock_dim,
            buy_cost_pct=[0.001] * self.stock_dim,  # 交易費用
            sell_cost_pct=[0.001] * self.stock_dim,  # 交易費用
            reward_scaling=1e-4,
            state_space=self.observation_shape,
            action_space=self.stock_dim,
            tech_indicator_list=self.tech_indicators,
            turbulence_threshold=None,
            risk_indicator_col='rsi14',
            print_verbosity=1,  # 不能為0，否則會導致零除錯誤
            day=0,
            initial=True
        )
        
        # 定義觀察和動作空間
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        print(f"創建環境包裝器 - 觀察空間: {self.observation_space}, 動作空間: {self.action_space}")
    
    def _process_observation(self, obs_tuple):
        """處理從FinRL環境返回的觀察"""
        if isinstance(obs_tuple, tuple) and len(obs_tuple) > 0:
            obs_list = obs_tuple[0]
            return np.array(obs_list, dtype=np.float32)
        else:
            return np.zeros(self.observation_shape, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """重置環境並返回初始觀察"""
        try:
            reset_result = self.env.reset(seed=seed)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs_tuple, info = reset_result
            else:
                obs_tuple = reset_result
                info = {}
        except TypeError:
            obs_tuple = self.env.reset()
            info = {}
        
        processed_obs = self._process_observation(obs_tuple)
        return processed_obs, info
    
    def step(self, action):
        """執行動作並返回下一個觀察"""
        action_np = np.array(action, dtype=np.float32)
        
        try:
            step_result = self.env.step(action_np)
            
            if len(step_result) == 5:
                obs_tuple, reward, terminated, truncated, info = step_result
                done = terminated
            elif len(step_result) == 4:
                obs_tuple, reward, done, info = step_result
                truncated = False
            else:
                raise ValueError(f"Unexpected number of return values from step(): {len(step_result)}")
            
            processed_obs = self._process_observation(obs_tuple)
            return processed_obs, reward, done, truncated, info
            
        except Exception as e:
            print(f"步驟執行錯誤: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.observation_shape), 0, True, False, {}


class PricePredictionModel:
    """使用FinRL進行價格預測的類"""
    
    def __init__(self, model_dir='trained_models'):
        """初始化模型"""
        self.model_dir = model_dir
        self.models = {}  # 不同時間窗口的模型字典
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化Binance數據獲取器
        self.data_fetcher = BinanceDataFetcher(BINANCE_API_KEY, BINANCE_API_SECRET)
    
    def train_model(self, symbol, time_window, lookback_days=60, timesteps=50000):
        """
        訓練特定時間窗口的模型
        
        參數:
            symbol (str): 交易對符號 (例如 'BTCUSDT')
            time_window (str): 時間窗口 ('30m', '1h', '1d')
            lookback_days (int): 用於訓練的歷史數據天數
            timesteps (int): 訓練步數
        """
        print(f"開始訓練 {symbol} {time_window} 模型...")
        
        # 獲取歷史數據
        df = self.data_fetcher.fetch_historical_data(symbol, time_window, lookback_days)
        
        # 添加技術指標
        df = TechnicalIndicators.add_indicators(df)
        
        # 創建環境
        env = FinRLEnvWrapper(df, time_window)
        
        # 創建模型 (使用PPO，通常比A2C表現更好)
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            device='cpu', 
            tensorboard_log=f"./logs/{symbol}_{time_window}",
            learning_rate=3e-4,  # 調整學習率
            gamma=0.99,  # 折扣因子
            ent_coef=0.01,  # 增加一點熵系數，鼓勵探索
            n_steps=2048,  # 增加每批次的步數
            batch_size=64  # 設置批次大小
        )
        
        # 訓練模型
        print(f"開始訓練 {symbol} {time_window} 模型，訓練步數: {timesteps}...")
        model.learn(total_timesteps=timesteps)
        
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{symbol}_{time_window}")
        model.save(model_path)
        print(f"模型已保存至: {model_path}")
        
        # 將模型添加到字典
        self.models[time_window] = model
        
        return model
    
    def load_model(self, symbol, time_window):
        """
        載入特定時間窗口的模型
        
        參數:
            symbol (str): 交易對符號 (例如 'BTCUSDT')
            time_window (str): 時間窗口 ('30m', '1h', '1d')
        """
        model_path = os.path.join(self.model_dir, f"{symbol}_{time_window}")
        
        if os.path.exists(model_path + ".zip"):
            print(f"載入模型: {model_path}")
            self.models[time_window] = PPO.load(model_path)
            return self.models[time_window]
        else:
            print(f"模型不存在: {model_path}，將進行訓練")
            return self.train_model(symbol, time_window)
    
    def prepare_prediction_env(self, symbol, time_window, lookback_days=5):
        """
        準備用於預測的環境
        
        參數:
            symbol (str): 交易對符號 (例如 'BTCUSDT')
            time_window (str): 時間窗口 ('30m', '1h', '1d')
            lookback_days (int): 用於預測的歷史數據天數
        """
        # 獲取最近的數據
        df = self.data_fetcher.fetch_historical_data(symbol, time_window, lookback_days)
        
        # 添加技術指標
        df = TechnicalIndicators.add_indicators(df)
        
        # 創建環境
        env = FinRLEnvWrapper(df, time_window)
        
        return env
    
    def predict_price_movement(self, symbol, time_window):
        """
        預測價格走勢
        
        參數:
            symbol (str): 交易對符號 (例如 'BTCUSDT')
            time_window (str): 時間窗口 ('30m', '1h', '1d')
            
        返回:
            dict: 預測結果
        """
        # 確保模型已加載
        if time_window not in self.models:
            self.load_model(symbol, time_window)
        
        # 獲取當前價格
        current_price = self.data_fetcher.get_current_price(symbol)
        
        # 準備環境
        env = self.prepare_prediction_env(symbol, time_window)
        
        # 重置環境並獲取觀察
        observation, _ = env.reset()
        
        # 使用模型預測動作
        try:
            action, _ = self.models[time_window].predict(observation, deterministic=True)
        except Exception as e:
            print(f"預測時發生錯誤: {e}")
            print("使用隨機猜測代替...")
            action = np.random.uniform(-0.5, 0.5, size=env.action_space.shape)
        
        # 獲取技術指標作為輔助決策依據
        try:
            # 獲取最近的數據
            recent_df = self.data_fetcher.fetch_historical_data(symbol, time_window, 1)
            recent_df = TechnicalIndicators.add_indicators(recent_df)
            latest_data = recent_df.iloc[-1]
            
            # 分析最近的技術指標
            ma_trend = 1 if latest_data['ma7'] > latest_data['ma25'] else -1
            rsi_signal = 1 if latest_data['rsi14'] > 50 else (-1 if latest_data['rsi14'] < 30 else 0)
            bb_position = (latest_data['close'] - latest_data['bb_lower']) / (latest_data['bb_upper'] - latest_data['bb_lower'])
            bb_signal = 1 if bb_position < 0.3 else (-1 if bb_position > 0.7 else 0)
            macd_signal = 1 if latest_data['macd'] > latest_data['macd_signal'] else -1
            
            # 綜合技術指標信號
            tech_signal = (ma_trend + rsi_signal + bb_signal + macd_signal) / 4
            
            # 結合模型預測和技術指標
            combined_signal = 0.7 * np.mean(action) + 0.3 * tech_signal
        except Exception as e:
            print(f"獲取技術指標時發生錯誤: {e}")
            combined_signal = np.mean(action)
        
        # 分析動作以確定預測的走勢
        if combined_signal > 0.1:
            prediction = "上漲"
            confidence = min(abs(combined_signal) * 100, 95)  # 限制最高置信度為95%
        elif combined_signal < -0.1:
            prediction = "下跌"
            confidence = min(abs(combined_signal) * 100, 95)
        else:
            prediction = "橫盤"
            confidence = 50 + min(abs(combined_signal) * 200, 25)  # 橫盤置信度範圍: 50-75%
            
        # 構建結果，包含更多分析信息
        result = {
            "symbol": symbol,
            "time_window": time_window,
            "current_price": current_price,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_signal": float(np.mean(action)),
            "technical_indicators": {
                "ma_trend": "金叉" if ma_trend > 0 else "死叉",
                "rsi_value": float(latest_data['rsi14']) if 'latest_data' in locals() else "未知",
                "bb_position": float(bb_position) if 'bb_position' in locals() else "未知",
                "macd_signal": "看漲" if 'macd_signal' in locals() and macd_signal > 0 else "看跌"
            }
        }
        
        return result


class BinancePricePredictionApp:
    """BTC/USDT價格預測應用程序"""
    
    def __init__(self):
        """初始化應用程序"""
        self.model = PricePredictionModel()
        self.symbol = "BTCUSDT"
    
    def train_models(self, lookback_days=60):
        """訓練所有時間窗口的模型"""
        for time_window in ['30m', '1h', '1d']:
            self.model.train_model(self.symbol, time_window, lookback_days)
    
    def predict(self, time_window=None):
        """
        預測價格走勢
        
        參數:
            time_window (str, optional): 時間窗口。如果為None，則預測所有時間窗口。
            
        返回:
            list: 預測結果列表
        """
        results = []
        
        if time_window:
            # 預測特定時間窗口
            result = self.model.predict_price_movement(self.symbol, time_window)
            results.append(result)
        else:
            # 預測所有時間窗口
            for tw in ['30m', '1h', '1d']:
                result = self.model.predict_price_movement(self.symbol, tw)
                results.append(result)
        
        return results


def main():
    """主函數"""
    print("初始化BTC/USDT價格預測系統...")
    app = BinancePricePredictionApp()
    
    # 檢查模型是否存在，不存在則訓練
    for time_window in ['30m', '1h', '1d']:
        model_path = os.path.join('trained_models', f"BTCUSDT_{time_window}.zip")
        if not os.path.exists(model_path):
            print(f"模型 {model_path} 不存在，開始訓練...")
            app.model.train_model("BTCUSDT", time_window)
    
    while True:
        print("\n請選擇操作:")
        print("1. 預測30分鐘價格走勢")
        print("2. 預測1小時價格走勢")
        print("3. 預測1天價格走勢")
        print("4. 預測所有時間窗口")
        print("5. 重新訓練模型")
        print("q. 退出")
        
        choice = input("請輸入選項: ")
        
        if choice == 'q':
            break
        
        try:
            if choice == '1':
                results = app.predict('30m')
            elif choice == '2':
                results = app.predict('1h')
            elif choice == '3':
                results = app.predict('1d')
            elif choice == '4':
                results = app.predict()
            elif choice == '5':
                lookback_days = int(input("請輸入用於訓練的歷史數據天數(默認60): ") or 60)
                app.train_models(lookback_days)
                continue
            else:
                print("無效選項，請重試")
                continue
            
            # 顯示預測結果
            print("\n預測結果:")
            for result in results:
                print(f"時間窗口: {result['time_window']}")
                print(f"當前價格: {result['current_price']}")
                print(f"預測走勢: {result['prediction']}")
                print(f"置信度: {result['confidence']:.2f}%")
                print(f"預測時間: {result['timestamp']}")
                print("------------------------")
            
        except Exception as e:
            print(f"發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    print("程序結束")


if __name__ == "__main__":
    main()