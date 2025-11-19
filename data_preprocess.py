import pandas as pd
import numpy as np
from etide.trading_data_access import DataLoaderEDP
from etide.feature_engineer import FeatureGenerator
from datetime import datetime, timedelta
# 初始化daas，fg
daas = DataLoaderEDP()
fg = FeatureGenerator()

# 定义获取电价函数
def get_price(st, et, province, node):
    if node == 'province':
        price_type = 'province'
        node = province
    else:
        price_type = 'node'
    true_price = daas.read_clearing_price(st, et, province, node, price_type=price_type)
    pred_dn = daas.read_forecast_price(st, et, province, node, 1, price_type=price_type)
    data = pd.merge(true_price, pred_dn, left_index=True, right_index=True, how='outer', suffixes=('', '_d+1预测'))
    data['实际价差'] = data['实时电价'] - data['日前电价']
    data['预测价差'] = data['实时电价_d+1预测'] - data['日前电价_d+1预测']
    return data

# 输入测试日期list，输出X_test与y_test
def data_prepare(pred_dates):
# 处理输入的日期list
    et = max(pred_dates) 
    st = min(pred_dates) - timedelta(days=2)
    real_st = min(pred_dates)
    et = et.strftime('%Y-%m-%d')
    st = st.strftime('%Y-%m-%d')
    real_st = real_st.strftime('%Y-%m-%d')

# 获取电价df
    df_price = get_price(st, et, 'shandong', 'hongtu110')
    
# 获取D-1供需df
    df_D1 = daas.read_grid_supply_demand(st,
                                et,
                                'shandong',
                                delay_days=1)
    df_D1.columns = [f"{col}_D-1" for col in df_D1.columns]
    
# 合并，返回总df
    df = pd.concat([df_price, df_D1], axis=1)
    df['竞价空间_D-1'] = df['系统负荷_D-1'] - df['新能源_D-1'] - df['联络线_D-1']

# 数据预处理
# 添加hour month特征
    df = fg.generate_features(df, feature_functions = ['hour','month'],cols = ['实际价差'])
    
# 部分特征log变换
    log_columns = ['新能源_D-1', '光伏_D-1', '地方电厂发电总加_D-1', '自备机组总加_D-1', '系统负荷_D-1', '联络线_D-1']
    for col in log_columns:
        df[col + '_log'] = np.log(df[col] + 1e-8)
    df.drop(columns=log_columns, inplace=True, errors='ignore')

# 添加预测价差shift特征
    df['预测价差_shift_192'] = df['预测价差'].shift(192)

# 排除label及label相关，生成特征list
    columns_to_exclude = (['实际价差'] +['实时电价']+['日前电价']+['风电_D-1'] +['试验机组总加_D-1'])
    features = df.columns.difference(columns_to_exclude)
    
 # 生成X_test与y_test
    X_test_raw = df[features]
    feature_order = ['hour', 'month', '光伏_D-1_log', '地方电厂发电总加_D-1_log', '实时电价_d+1预测',
       '抽蓄_D-1', '新能源_D-1_log', '日前电价_d+1预测', '竞价空间_D-1', '系统负荷_D-1_log',
       '联络线_D-1_log', '自备机组总加_D-1_log', '预测价差', '预测价差_shift_192']
    X_test = X_test_raw[feature_order]
    y_test = df['实际价差']
    X_test = X_test.loc[real_st:]
    y_test = y_test.loc[real_st:]

    return X_test, y_test
