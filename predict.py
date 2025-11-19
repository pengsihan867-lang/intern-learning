import pandas as pd
import joblib
from data_preprocess import data_prepare
import os

_cur_file = os.path.dirname(__file__)

def LGBM_predict_price_diff(dates):
    # 准备数据
    X_test, y_test = data_prepare(dates)

    # 模型路径
    median_model_path = os.path.abspath(os.path.join(_cur_file, 'model/model_median.pkl'))
    lower_model_path = os.path.abspath(os.path.join(_cur_file, 'model/model_lower.pkl'))
    upper_model_path = os.path.abspath(os.path.join(_cur_file, 'model/model_upper.pkl'))

    # 加载模型
    model_median = joblib.load(median_model_path)
    model_lower = joblib.load(lower_model_path)
    model_upper = joblib.load(upper_model_path)

    # 调用模型
    pred_median = model_median.predict(X_test)
    pred_upper = model_upper.predict(X_test)
    pred_lower = model_lower.predict(X_test)

    # 后处理：保证下分位数预测结果低于上分位数预测结果
    mask = pred_lower > pred_upper
    pred_lower[mask], pred_upper[mask] = pred_upper[mask], pred_lower[mask]

    # 输出预测结果
    df_pred = pd.DataFrame({
        '5%分位数预测': pred_lower,
        '50%分位数预测': pred_median,
        '95%分位数预测': pred_upper
    }, index=X_test.index)
    
    return df_pred