import math

# 定義函數來計算 decay_rate
def calculate_decay_rate(initial_epsilon, min_epsilon, updates):
    # 確保輸入有效
    if initial_epsilon <= 0 or min_epsilon <= 0 or min_epsilon >= initial_epsilon or updates <= 0:
        raise ValueError("輸入參數無效")

    # 使用對數公式計算 decay_rate
    ln_min_epsilon = math.log(min_epsilon)
    
    # 推導 decay_rate
    exponent = ln_min_epsilon / updates
    decay_rate = math.exp(exponent)
    
    return decay_rate

if __name__ == "__main__":
    
    # 測試案例
    # 假設 initial_epsilon = 1.0, min_epsilon = 0.01, updates 為你提供的數值
    initial_epsilon = 1.0
    min_epsilon = 0.01

    # 案例 
    updates = 5133 * 30 # (5165 - 32) * 想要在第幾個episode到達最小值
    decay_rate1 = calculate_decay_rate(initial_epsilon, min_epsilon, updates)

    # 印出結果
    print(f"當 initial_epsilon = {initial_epsilon}, min_epsilon = {min_epsilon}, updates = {updates} 時，")
    print(f"推算的 decay_rate 約為: {decay_rate1:.6f}")
