import numpy as np

def kalman_filter(prices, F, H, Q, R):
    x_est = prices[0]
    P_est = 1
    filtered = np.zeros(len(prices))

    for i,price in enumerate(prices):
        x_pred = F * x_est
        P_pred = F * P_est * F + Q
        K_t = P_pred / (H*P_pred*H+R)
        x_est = x_pred + K_t * (price - H * x_pred)
        P_est = (1-K_t*H) * P_pred
        filtered[i] = x_est

    return filtered