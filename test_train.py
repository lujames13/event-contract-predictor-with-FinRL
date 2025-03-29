"""
FinRL 測試程式 - 最小可行的強化學習交易範例

本程式演示如何使用 FinRL 庫建立一個簡單的交易環境，並使用 Stable-Baselines3 的 A2C 演算法進行訓練。
程式中包含了處理新版 FinRL 與 Gymnasium API 整合的關鍵發現和最佳實踐。

主要發現與注意事項:
1. 觀察空間形狀: FinRL 的 StockTradingEnv 返回的觀察是一個元組 ([a, b, c, d, e], {})，其中第一個元素是
   包含 5 個值的列表 (總資產, 收盤價, 持倉, 技術指標1, 技術指標2)。

2. 交易成本參數: buy_cost_pct 和 sell_cost_pct 必須是與股票數量相同長度的數組，而不是單一浮點數。

3. API 兼容性: 需要處理新舊版本的 Gymnasium API 差異:
   - reset(): 新API 返回 (obs, info)，舊API 只返回 obs
   - step(): 新API 返回 (obs, reward, terminated, truncated, info)，舊API 返回 (obs, reward, done, info)

4. 動作格式: 傳入 step() 的動作必須是 numpy 數組，不能是列表，否則會出現 'list' object has no attribute 'astype' 錯誤。

5. 包裝器設計: 建立一個適當的包裝器來處理 FinRL 環境與 Stable-Baselines3 之間的接口差異是關鍵。
"""

import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from stable_baselines3 import A2C
import gymnasium as gym

# 載入環境變數
load_dotenv()

# 使用小型數據集進行測試
def get_test_data():
    """
    創建一個用於測試的合成數據集
    
    返回:
        pd.DataFrame: 包含股價和技術指標的數據框架
    """
    # 創建一個簡單的假數據集
    dates = pd.date_range(start='2023-01-01', periods=500, freq='d')
    
    # 創建數據
    data = []
    for i, date in enumerate(dates):
        close_price = np.random.randn() * 100 + 25000
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'tic': 'BTCUSDT',
            'open': close_price + np.random.randn() * 50,
            'high': close_price + abs(np.random.randn() * 100),
            'low': close_price - abs(np.random.randn() * 100),
            'close': close_price,
            'volume': abs(np.random.randn() * 1000 + 10000),
            'day': i
        })
    
    df = pd.DataFrame(data)
    
    # 添加技術指標
    df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['rsi_14'] = 50 + np.random.randn(500) * 10
    
    # 確保所有列都是數值型態
    for col in df.columns:
        if col not in ['date', 'tic', 'day']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 確保沒有NaN值
    df = df.ffill().bfill()
    
    print("數據框架頭部:")
    print(df.head())
    print("\n數據框架列:")
    print(df.columns.tolist())
    
    return df

# 創建一個完善的包裝器來處理FinRL環境
class FinRLEnvWrapper(gym.Env):
    """
    FinRL環境的包裝器，用於處理與Stable-Baselines3的接口兼容性
    
    這個包裝器解決了以下問題:
    1. 觀察空間形狀不匹配
    2. API差異 (新舊版本的Gymnasium)
    3. 數據類型轉換
    """
    def __init__(self, df):
        """
        初始化環境包裝器
        
        參數:
            df (pd.DataFrame): 股價數據框架
        """
        super(FinRLEnvWrapper, self).__init__()
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        
        # 獲取基本配置
        self.stock_dim = len(df['tic'].unique())
        self.tech_indicators = ['ma10', 'rsi_14']
        
        # 根據測試，FinRL返回的觀察是一個元組，其中第一個元素包含5個值
        self.observation_shape = 5
        
        # 創建原始環境 - 注意buy_cost_pct和sell_cost_pct必須是數組
        self.env = StockTradingEnv(
            df=df,
            stock_dim=self.stock_dim,
            hmax=100,
            initial_amount=1000,
            num_stock_shares=[0] * self.stock_dim,
            buy_cost_pct=[0.001] * self.stock_dim,  # 必須是數組形式，不能是浮點數
            sell_cost_pct=[0.001] * self.stock_dim,  # 必須是數組形式，不能是浮點數
            reward_scaling=1e-3,
            state_space=self.observation_shape,  # 根據FinRL實現設置正確的狀態空間
            action_space=self.stock_dim,
            tech_indicator_list=self.tech_indicators,
            turbulence_threshold=None,
            risk_indicator_col='rsi_14',
            print_verbosity=1,
            day=0,
            initial=True
        )
        
        # 定義觀察和動作空間，確保與FinRL環境匹配
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        print(f"創建環境包裝器 - 觀察空間: {self.observation_space}, 動作空間: {self.action_space}")
    
    def _process_observation(self, obs_tuple):
        """
        處理從FinRL環境返回的觀察
        
        FinRL環境返回的觀察是一個元組 ([a, b, c, d, e], {})，
        我們需要提取第一個元素並轉換為numpy數組
        
        參數:
            obs_tuple: FinRL環境返回的原始觀察
            
        返回:
            np.array: 處理後的觀察
        """
        if isinstance(obs_tuple, tuple) and len(obs_tuple) > 0:
            obs_list = obs_tuple[0]
            # 確保obs_list是numpy數組並且形狀正確
            return np.array(obs_list, dtype=np.float32)
        else:
            # 如果觀察不是預期的元組，返回默認值
            return np.zeros(self.observation_shape, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        重置環境並返回初始觀察
        
        處理新舊版本Gymnasium API的差異
        
        參數:
            seed: 隨機種子
            options: 其他選項
            
        返回:
            tuple: (觀察, 信息字典)
        """
        # 嘗試使用新的reset API
        try:
            reset_result = self.env.reset(seed=seed)
            # 檢查reset_result是否是新API格式(obs, info)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs_tuple, info = reset_result
            else:
                # 舊API僅返回obs
                obs_tuple = reset_result
                info = {}
        except TypeError:
            # 舊API不支持seed參數
            obs_tuple = self.env.reset()
            info = {}
        
        # 處理並返回觀察
        processed_obs = self._process_observation(obs_tuple)
        print(f"Reset - 原始觀察: {obs_tuple}")
        print(f"Reset - 處理後觀察: {processed_obs}, 形狀: {processed_obs.shape}")
        
        return processed_obs, info
    
    def step(self, action):
        """
        執行動作並返回下一個觀察
        
        處理新舊版本Gymnasium API的差異
        
        參數:
            action: 要執行的動作
            
        返回:
            tuple: (觀察, 獎勵, 終止標誌, 截斷標誌, 信息字典)
        """
        # 將動作轉換為numpy數組 - 這一步很重要，否則會出現'list' object has no attribute 'astype'錯誤
        action_np = np.array(action, dtype=np.float32)
        
        # 執行動作
        try:
            # 嘗試使用新的API
            step_result = self.env.step(action_np)
            
            # 檢查返回的值數量
            if len(step_result) == 5:
                # 新API: obs, reward, terminated, truncated, info
                obs_tuple, reward, terminated, truncated, info = step_result
                done = terminated
            elif len(step_result) == 4:
                # 舊API: obs, reward, done, info
                obs_tuple, reward, done, info = step_result
                truncated = False
            else:
                raise ValueError(f"Unexpected number of return values from step(): {len(step_result)}")
            
            # 處理觀察
            processed_obs = self._process_observation(obs_tuple)
            
            # 打印調試信息
            if np.random.random() < 0.01:  # 只打印1%的步驟
                print(f"Step - 動作: {action}, 獎勵: {reward}")
                print(f"Step - 原始觀察: {obs_tuple}")
                print(f"Step - 處理後觀察: {processed_obs}")
            
            return processed_obs, reward, done, truncated, info
        except Exception as e:
            print(f"步驟執行錯誤: {e}")
            import traceback
            traceback.print_exc()
            # 返回默認值
            return np.zeros(self.observation_shape), 0, True, False, {}

# 主程序
def main():
    """
    主程序入口
    
    創建環境、訓練模型並保存
    """
    # 創建保存目錄
    os.makedirs('trained_models', exist_ok=True)
    
    # 獲取測試數據
    df = get_test_data()
    print(f"測試數據大小: {df.shape}")
    
    # 創建環境
    try:
        # 使用包裝器創建環境
        env = FinRLEnvWrapper(df)
        
        # 創建模型
        model = A2C('MlpPolicy', env, verbose=1, device='cpu')
        
        # 訓練模型
        print("開始訓練模型...")
        model.learn(total_timesteps=10000)
        print("模型訓練完成")
        
        # 保存模型
        model_path = os.path.join('trained_models', f"test_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save(model_path)
        print(f"模型已保存至: {model_path}")
    
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()