from predict import LGBM_predict_price_diff
import pandas as pd

if __name__ == '__main__':
    df_pred = LGBM_predict_price_diff(pd.date_range(start='2025-07-01', end='2025-07-28', freq='D').tolist())
    df_pred.to_csv('pred_price_data_0701-0728.csv', encoding='utf-8-sig')