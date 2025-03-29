import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import time
import gym
import torch
import argparse
from dotenv import load_dotenv

# 有選擇性地導入FinRL組件，避免不必要的依賴
try:
    from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
except ImportError:
    print("無法導入 stable_baselines3，請執行: pip install stable-baselines3")

# 我們將實現自己的 StockTradingEnv 和 DRLAgent 類，而不依賴 FinRL

# Binance API
from binance.client import Client
from binance.exceptions import BinanceAPIException

# 載入環境變數
load_dotenv()

# 設定配置
class Config:
    START_DATE = "2020-01-01"
    END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    TICKER_LIST = ["BTC/USDT"]
    
    # 模型參數
    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    }
    DDPG_PARAMS = {"buffer_size": 10000, "learning_rate": 0.0005}
    TD3_PARAMS = {"buffer_size": 10000, "learning_rate": 0.0005}
    SAC_PARAMS = {"buffer_size": 10000, "learning_rate": 0.0001, "batch_size": 64}
    
    # 指定使用的模型
    TRAINED_MODEL_DIR = "trained_models"
    RESULTS_DIR = "results"
    TENSORBOARD_LOG_DIR = "tensorboard_log"
    MODELS = ["a2c", "ppo", "ddpg", "td3", "sac"]  # 可選模型
    
    # Binance 設定 - 從環境變數中獲取
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# 數據獲取與處理類
class CryptoDataProcessor:
    def __init__(self, config):
        self.config = config
        self.binance_client = Client(config.API_KEY, config.API_SECRET)
        
    def download_data(self, symbol="BTCUSDT", interval="1h", start_date=None, end_date=None):
        """從Binance下載歷史K線數據"""
        try:
            if start_date is None:
                start_date = self.config.START_DATE
            if end_date is None:
                end_date = self.config.END_DATE
                
            # 轉換日期格式
            start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
            print(f"下載 {symbol} 的 {interval} 數據，從 {start_date} 到 {end_date}")
            
            # 獲取K線數據
            klines = self.binance_client.get_historical_klines(
                symbol, 
                interval, 
                start_date_obj.strftime("%d %b, %Y"),
                end_date_obj.strftime("%d %b, %Y")
            )
            
            print(f"成功下載 {len(klines)} 條K線數據")
            
            # 轉換為DataFrame
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 數據處理
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            # 轉換數值類型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = data[col].astype(float)
                
            # 添加資產標識
            data['tic'] = symbol
            
            return data
        
        except BinanceAPIException as e:
            print(f"下載數據時出錯: {e}")
            return None
    
    def feature_engineering(self, df):
        """特徵工程處理"""
        print("執行特徵工程...")
        
        # 檢查數據大小
        print(f"原始數據大小: {df.shape}")
        
        # 複製數據以避免修改原始數據
        df = df.copy()
        
        # 基本技術指標
        df['returns'] = df['close'].pct_change().fillna(0)
        
        # 使用clip來限制極端值
        df['returns'] = df['returns'].clip(-0.5, 0.5)
        
        # 安全地計算log_returns
        eps = 1e-8  # 小數值避免log(0)
        price_ratio = df['close'] / df['close'].shift(1).replace(0, eps)
        df['log_returns'] = np.log(price_ratio.clip(eps, 1/eps)).fillna(0)
        
        # 價格變動
        df['price_change'] = (df['close'] - df['open']).fillna(0)
        
        # 安全地計算百分比變動
        open_prices = df['open'].replace(0, eps)
        df['price_change_pct'] = ((df['close'] - df['open']) / open_prices).fillna(0).clip(-1, 1)
        
        # 移動平均線 - 使用min_periods確保不會產生NaN
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['ma200'] = df['close'].rolling(window=200, min_periods=1).mean()
        
        # 安全地計算與MA的距離
        for ma in ['ma5', 'ma10', 'ma20']:
            ma_values = df[ma].replace(0, eps)
            df[f'{ma}_dist'] = ((df['close'] - df[ma]) / ma_values).fillna(0).clip(-1, 1)
        
        # 波動性指標 - 使用min_periods避免NaN
        df['volatility_5'] = df['returns'].rolling(window=5, min_periods=1).std().fillna(0)
        df['volatility_10'] = df['returns'].rolling(window=10, min_periods=1).std().fillna(0)
        df['volatility_20'] = df['returns'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # RSI - 避免除以零
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean().fillna(0)
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean().fillna(0)
        
        # 避免除以零
        rs = np.where(loss == 0, 100, gain / (loss + eps))
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50)  # 使用中性值填充
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp26 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = (exp12 - exp26).fillna(0)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean().fillna(0)
        df['macd_hist'] = (df['macd'] - df['macd_signal']).fillna(0)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean().fillna(df['close'])
        df['bb_std'] = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # 安全地計算BB寬度
        middle_values = df['bb_middle'].replace(0, eps)
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / middle_values).fillna(0).clip(0, 2)
        
        # 成交量指標
        df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean().fillna(df['volume'])
        df['volume_ma10'] = df['volume'].rolling(window=10, min_periods=1).mean().fillna(df['volume'])
        df['volume_change'] = df['volume'].pct_change().fillna(0).clip(-1, 1)
        
        # 標準化或縮放特徵以改善模型訓練
        # 使用z-score標準化，但限制極端值
        for col in df.columns:
            if col not in ['timestamp', 'tic']:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = ((df[col] - mean) / std).clip(-3, 3)
                else:
                    df[col] = 0  # 如果標準差為0，將特徵設為0
        
        # 檢查並處理任何剩餘的NaN值
        df.fillna(0, inplace=True)
        
        # 確保沒有inf值
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # 最終數據檢查
        print(f"特徵工程後數據大小: {df.shape}")
        print(f"檢查NaN值: {df.isna().sum().sum()}")
        print(f"檢查Inf值: {np.isinf(df.values).sum()}")
        
        return df
        
    def get_processed_data(self, symbol="BTCUSDT", interval="1h"):
        """獲取處理後的數據"""
        # 下載數據
        df = self.download_data(symbol, interval)
        if df is None or len(df) == 0:
            print("無法獲取數據或數據為空")
            return None
            
        # 特徵工程
        processed_df = self.feature_engineering(df)
        
        return processed_df
        
    def split_data(self, df, train_ratio=0.8):
        """將數據分割為訓練集和測試集"""
        if df is None or len(df) == 0:
            return None, None
            
        split_point = int(len(df) * train_ratio)
        train_data = df[:split_point]
        test_data = df[split_point:]
        
        print(f"訓練集大小: {train_data.shape}, 測試集大小: {test_data.shape}")
        
        return train_data, test_data

# 交易環境創建 (使用FinRL)
class CryptoTradingEnv:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.df = data  # 提供對DataFrame的直接訪問
        self.window_size = 20  # 定義觀察窗口大小供評估時使用
        self.total_assets = 1000  # 初始資產
        
    def create_env(self, train=True):
        """創建交易環境"""
        # 檢查數據是否有問題
        if self.data is None or len(self.data) == 0:
            print("無法創建環境: 數據為空")
            return None
            
        # 檢查特徵名稱
        tech_indicator_list = self._get_feature_list()
        for tech in tech_indicator_list:
            if tech not in self.data.columns:
                print(f"警告: 特徵 '{tech}' 不在數據集中")
                
        # 額外清理數據以避免NaN/Inf問題
        df = self.data.copy()
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        
        print(f"創建{'訓練' if train else '測試'}環境，數據形狀: {df.shape}")
        
        try:
            # 強制緩存DataFrame來避免潛在的錯誤
            df_cached = df.copy()
            
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
            env = StockTradingEnv(
                df=df_cached,
                initial_amount=1000,
                commission=0.001,  # Binance的交易手續費
                reward_scaling=1e-4,
                tech_indicator_list=tech_indicator_list,
                turbulence_threshold=None,
                risk_indicator_col=None,
                print_verbosity=1  # 啟用詳細日誌
            )
            
            # 包裝環境以增加監控
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
            
            print("環境創建成功")
            return env
            
        except Exception as e:
            print(f"創建環境時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_feature_list(self):
        """獲取特徵列表"""
        # 檢查所有特徵是否存在
        all_features = [
            'returns', 'log_returns', 'price_change', 'price_change_pct',
            'ma5', 'ma10', 'ma20', 'ma50', 'ma200',
            'ma5_dist', 'ma10_dist', 'ma20_dist',
            'volatility_5', 'volatility_10', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width',
            'volume_ma5', 'volume_ma10', 'volume_change'
        ]
        
        # 只使用數據中實際存在的特徵
        if self.data is not None:
            available_features = [f for f in all_features if f in self.data.columns]
            return available_features
        
        return all_features

# 模型訓練類
class ModelTrainer:
    def __init__(self, config, env_creator):
        self.config = config
        self.env_creator = env_creator
        
        # 創建保存目錄
        if not os.path.exists(config.TRAINED_MODEL_DIR):
            os.makedirs(config.TRAINED_MODEL_DIR)
        if not os.path.exists(config.RESULTS_DIR):
            os.makedirs(config.RESULTS_DIR)
        if not os.path.exists(config.TENSORBOARD_LOG_DIR):
            os.makedirs(config.TENSORBOARD_LOG_DIR)
            
    def train_model(self, model_name, train_env, total_timesteps=100000):
        """訓練指定的模型"""
        # 強制使用CPU訓練以避免CUDA錯誤
        device = 'cpu'
        
        print(f"使用 {device} 訓練 {model_name} 模型，總步數: {total_timesteps}")
        
        # 設置模型參數
        if model_name == "a2c":
            params = dict(self.config.A2C_PARAMS)
            params['device'] = device
            model = A2C(
                policy="MlpPolicy",
                env=train_env,
                **params,
                tensorboard_log=self.config.TENSORBOARD_LOG_DIR
            )
        elif model_name == "ppo":
            params = dict(self.config.PPO_PARAMS)
            params['device'] = device
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                **params,
                tensorboard_log=self.config.TENSORBOARD_LOG_DIR
            )
        elif model_name == "ddpg":
            params = dict(self.config.DDPG_PARAMS)
            params['device'] = device
            model = DDPG(
                policy="MlpPolicy",
                env=train_env,
                **params,
                tensorboard_log=self.config.TENSORBOARD_LOG_DIR
            )
        elif model_name == "td3":
            params = dict(self.config.TD3_PARAMS)
            params['device'] = device
            model = TD3(
                policy="MlpPolicy",
                env=train_env,
                **params,
                tensorboard_log=self.config.TENSORBOARD_LOG_DIR
            )
        elif model_name == "sac":
            params = dict(self.config.SAC_PARAMS)
            params['device'] = device
            model = SAC(
                policy="MlpPolicy",
                env=train_env,
                **params,
                tensorboard_log=self.config.TENSORBOARD_LOG_DIR
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")
            
        # 訓練模型
        try:
            model.learn(total_timesteps=total_timesteps)
            print("模型訓練完成")
        except Exception as e:
            print(f"訓練過程中出現錯誤: {e}")
            # 嘗試減少批次大小並重新訓練
            if hasattr(model, 'batch_size') and model.batch_size > 32:
                print("嘗試減少批次大小並重新訓練...")
                model.batch_size = 32
                model.learn(total_timesteps=total_timesteps)
            else:
                raise
        
        # 保存模型
        model_save_path = os.path.join(
            self.config.TRAINED_MODEL_DIR,
            f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model.save(model_save_path)
        print(f"模型已保存至: {model_save_path}")
        
        return model, model_save_path
    
    def evaluate_model(self, model, test_env):
        """評估模型性能"""
        print("開始評估模型...")
        
        # 使用模型進行預測
        observations = []
        actions = []
        rewards = []
        account_values = []
        
        obs = test_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            observations.append(obs)
            actions.append(action)
            
            obs, reward, done, info = test_env.step(action)
            rewards.append(reward)
            account_values.append(test_env.total_assets)
            episode_reward += reward
            
        print(f"評估完成，總獎勵: {episode_reward:.2f}")
        
        # 創建評估報告
        dates = pd.date_range(
            start=test_env.df.index[test_env.window_size],
            periods=len(account_values),
            freq='H'  # 假設時間間隔為1小時，根據實際情況調整
        )
        
        df_account_value = pd.DataFrame({
            'account_value': account_values,
            'date': dates
        })
        df_account_value.set_index('date', inplace=True)
        
        df_actions = pd.DataFrame({
            'action': [a[0] for a in actions],
            'date': dates
        })
        df_actions.set_index('date', inplace=True)
        
        # 保存回測結果
        results_path = os.path.join(
            self.config.RESULTS_DIR,
            f"backtest_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_account_value.to_csv(results_path)
        print(f"回測結果已保存至: {results_path}")
        
        return df_account_value, df_actions
        
    def backtest_stats(self, df_account_value):
        """計算回測統計數據"""
        print("計算回測統計數據...")
        
        df = df_account_value.copy()
        df['daily_return'] = df['account_value'].pct_change().fillna(0)
        
        stats = {}
        stats['初始資產'] = df['account_value'].iloc[0]
        stats['最終資產'] = df['account_value'].iloc[-1]
        stats['總回報 (%)'] = ((df['account_value'].iloc[-1] / df['account_value'].iloc[0]) - 1) * 100
        
        # 計算年化回報率（假設交易日為252天）
        trading_days = len(df) / 252
        if trading_days > 0:
            stats['年化回報率 (%)'] = ((1 + stats['總回報 (%)'] / 100) ** (1 / trading_days) - 1) * 100
        else:
            stats['年化回報率 (%)'] = 0
            
        # 計算夏普比率（假設無風險利率為0）
        std = np.std(df['daily_return'])
        if std > 0:
            stats['夏普比率'] = np.mean(df['daily_return']) / std * np.sqrt(252)
        else:
            stats['夏普比率'] = 0
            
        # 計算波動率
        stats['波動率 (%)'] = std * np.sqrt(252) * 100
        
        # 計算最大回撤
        cumulative_returns = (1 + df['daily_return']).cumprod()
        max_return = cumulative_returns.cummax()
        drawdown = (cumulative_returns / max_return - 1)
        stats['最大回撤 (%)'] = drawdown.min() * 100
        
        # 計算勝率
        winning_days = (df['daily_return'] > 0).sum()
        total_days = len(df)
        if total_days > 0:
            stats['勝率 (%)'] = winning_days / total_days * 100
        else:
            stats['勝率 (%)'] = 0
            
        # 計算盈虧比
        gains = df.loc[df['daily_return'] > 0, 'daily_return'].mean()
        losses = abs(df.loc[df['daily_return'] < 0, 'daily_return'].mean())
        if losses > 0:
            stats['盈虧比'] = gains / losses
        else:
            stats['盈虧比'] = float('inf')
            
        print("回測統計計算完成")
        return stats

# 預測服務類
class PredictionService:
    def __init__(self, config, data_processor, model_path):
        self.config = config
        self.data_processor = data_processor
        self.model = self._load_model(model_path)
        self.binance_client = Client(config.API_KEY, config.API_SECRET)
    
    def _load_model(self, model_path):
        """加載訓練好的模型"""
        # 檢查模型類型並加載
        if "a2c" in model_path:
            model = A2C.load(model_path)
        elif "ppo" in model_path:
            model = PPO.load(model_path)
        elif "ddpg" in model_path:
            model = DDPG.load(model_path)
        elif "td3" in model_path:
            model = TD3.load(model_path)
        elif "sac" in model_path:
            model = SAC.load(model_path)
        else:
            raise ValueError(f"無法識別模型類型: {model_path}")
            
        return model
    
    def get_current_price(self, symbol="BTCUSDT"):
        """獲取當前價格"""
        ticker = self.binance_client.get_ticker(symbol=symbol)
        return float(ticker['lastPrice'])
    
    def predict_price_movement(self, timeframe="1h", periods=1):
        """預測價格走勢
        timeframe: 時間週期，如 '30m', '1h', '1d'
        periods: 預測多少個時間週期後的價格
        """
        # 獲取最新數據
        latest_data = self.data_processor.get_processed_data(
            symbol="BTCUSDT", 
            interval=timeframe
        )
        
        if latest_data is None:
            return {"error": "無法獲取數據"}
            
        # 創建預測環境
        test_env = CryptoTradingEnv(self.config, latest_data)
        
        # 使用模型進行預測
        observations = []
        actions = []
        rewards = []
        
        obs = test_env.reset()
        done = False
        steps = 0
        
        # 只預測指定的期數
        while not done and steps < periods:
            action, _states = self.model.predict(obs)
            observations.append(obs)
            actions.append(action)
            
            obs, reward, done, info = test_env.step(action)
            rewards.append(reward)
            steps += 1
        
        # 獲取當前價格
        current_price = self.get_current_price()
        
        # 分析預測結果
        # 根據模型的動作來判斷價格走勢 (>0.5表示買入，<-0.5表示賣出)
        action_values = [a[0] for a in actions]
        
        # 計算預測趨勢
        if len(action_values) == 0:
            prediction = "無法預測"
            confidence = 0.0
        else:
            avg_action = sum(action_values) / len(action_values)
            
            if avg_action > 0.2:  # 傾向買入
                prediction = "上升"
                confidence = min((avg_action + 1) / 2, 1.0)  # 轉換到0-1範圍
            elif avg_action < -0.2:  # 傾向賣出
                prediction = "下降"
                confidence = min((abs(avg_action) + 1) / 2, 1.0)
            else:
                prediction = "維持不變"
                confidence = 0.5
            
        return {
            "current_price": current_price,
            "timeframe": timeframe,
            "periods": periods,
            "prediction": prediction,
            "confidence": confidence,
            "actions": action_values,  # 返回詳細的動作值
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
def main():
    # 確保環境變數已載入
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("警告: 找不到 Binance API 憑證。請確保 .env 文件包含 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        print("範例 .env 檔案內容:")
        print("BINANCE_API_KEY=your_api_key_here")
        print("BINANCE_API_SECRET=your_api_secret_here")
        return
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='BTC/USDT 價格預測')
    parser.add_argument('--train', action='store_true', help='訓練新模型')
    parser.add_argument('--model', type=str, default='ppo', help='選擇模型類型(a2c, ppo, ddpg, td3, sac)')
    parser.add_argument('--timesteps', type=int, default=100000, help='訓練總步數')
    parser.add_argument('--predict', action='store_true', help='進行預測')
    parser.add_argument('--timeframe', type=str, default='1h', help='預測時間週期(30m, 1h, 1d)')
    parser.add_argument('--periods', type=int, default=1, help='預測週期數')
    
    args = parser.parse_args()
    
    # 初始化設置
    config = Config()
    data_processor = CryptoDataProcessor(config)
    
    if args.train:
        print(f"正在訓練 {args.model} 模型...")
        
        # 獲取訓練數據
        df = data_processor.get_processed_data(interval=args.timeframe)
        train_data, test_data = data_processor.split_data(df)
        
        # 創建環境
        train_env_creator = CryptoTradingEnv(config, train_data)
        train_env = train_env_creator.create_env(train=True)
        
        # 訓練模型
        trainer = ModelTrainer(config, train_env_creator)
        trained_model, model_path = trainer.train_model(model_name=args.model, train_env=train_env, total_timesteps=args.timesteps)
        
        print(f"模型訓練完成，保存至: {model_path}")
        
        # 評估模型
        test_env_creator = CryptoTradingEnv(config, test_data)
        test_env = test_env_creator.create_env(train=False)
        
        df_account_value, df_actions = trainer.evaluate_model(model=trained_model, test_env=test_env)
        
        # 計算性能指標
        perf_stats_all = backtest_stats(df_account_value)
        print("模型性能評估:")
        print(perf_stats_all)
        
    if args.predict:
        # 使用最近訓練的模型或指定的模型
        if args.train:
            # 使用剛訓練的模型
            pass  # model_path已經在training階段設置好了
        else:
            # 尋找最新訓練的模型
            model_files = [f for f in os.listdir(config.TRAINED_MODEL_DIR) if f.startswith(args.model)]
            if not model_files:
                print(f"找不到 {args.model} 的模型文件，請先訓練模型")
                return
            
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model)
        
        # 創建預測服務
        prediction_service = PredictionService(config, data_processor, model_path)
        
        # 進行預測
        result = prediction_service.predict_price_movement(timeframe=args.timeframe, periods=args.periods)
        
        print("預測結果:")
        for key, value in result.items():
            print(f"{key}: {value}")

# 創建API服務
def create_api_service():
    """創建一個簡單的API服務，當收到請求時進行預測"""
    from flask import Flask, request, jsonify
    
    # 確保環境變數已載入
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("警告: 找不到 Binance API 憑證。請確保 .env 文件包含 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        print("範例 .env 檔案內容:")
        print("BINANCE_API_KEY=your_api_key_here")
        print("BINANCE_API_SECRET=your_api_secret_here")
        return
    
    app = Flask(__name__)
    
    # 初始化設置
    config = Config()
    data_processor = CryptoDataProcessor(config)
    
    # 尋找最新訓練的模型
    model_files = []
    for model_type in config.MODELS:
        files = [f for f in os.listdir(config.TRAINED_MODEL_DIR) if f.startswith(model_type)]
        if files:
            model_files.extend([(model_type, f) for f in files])
    
    if not model_files:
        print("找不到訓練好的模型，請先訓練模型")
        return
    
    # 使用最新的模型
    latest_model = sorted(model_files, key=lambda x: x[1])[-1]
    model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model[1])
    
    # 創建預測服務
    prediction_service = PredictionService(config, data_processor, model_path)
    
    @app.route('/predict', methods=['GET'])
    def predict():
        timeframe = request.args.get('timeframe', '1h')
        periods = int(request.args.get('periods', 1))
        
        result = prediction_service.predict_price_movement(timeframe=timeframe, periods=periods)
        
        return jsonify(result)
    
    # 啟動API服務
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()