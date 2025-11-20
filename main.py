from power_opt import PowerOpt
import pandas as pd
import numpy as np
from util import data_prepare, reward_calculate, generate_and_reduce_quantile_scenarios,\
    generate_and_reduce_laplace_scenarios, reduce_scenarios_ffs, reduce_scenarios_hierarchical, reduce_scenarios_heuristic

if __name__ == '__main__':
    # 初始化poweropt类
    poweropt = PowerOpt()

    # 准备相关参数
    max_power = 49.3
    station = 'zhangjiachan'

    # 准备输入数据
    data = data_prepare(start_date='2025-01-01', end_date='2025-07-31', station = station)

    # 生成与削减场景
    reduced_scenarios, scenarios_probs = generate_and_reduce_quantile_scenarios(data['df_full_quantiles'],data['pred_da_prices'],500,30)

    # 调用优化函数
    opt_powers= poweropt.optimize(
        max_power = max_power,
        site_type = 'wind',
        is_join_market = 1,
        pred_powers = data['pred_powers'],
        pred_da_limit_ratio =0,
        pred_da_prices = data['pred_da_prices'],
        pred_id_prices = data['pred_id_prices'],
        longterm_powers = None,
        longterm_prices = None,
        is_user_defined = None,
        adjust_max = None,
        adjust_min = None,
        is_robust=0,
        pred_spread=data['pred_spread'],
        pred_spread_lower=data['pred_spread_lower'],
        pred_spread_upper=data['pred_spread_upper'],
        is_chance_constr=0,
        is_scenario=0,
        scenarios = reduced_scenarios,
        scenarios_probs = scenarios_probs
    )

    # 计算当前收益
    reward = reward_calculate(
            dayahead_powers = opt_powers,  # 调整后日前申报量
            real_powers = data['real_powers'],  # 实发功率，真实值
            dayahead_price = data['dayahead_price'],  # 日前电价，真实值
            realtime_price = data['realtime_price'],  # 实时电价，真实值
            assess_ratio = data['params']['ASSESS_RATIO'],  # 偏差考核比例
            assess_price = data['params']['ASSESS_PRICE'],  # 偏差考核基准价格
            error_tolerance =data['params']['ERROR_TOLERANCE']['wind'],  # 新能源获利回收中偏差允许范围
            max_power= max_power, # 场站容量，用于功率准确率计算
            set_time_granularity = 15 # 设置计算颗粒度
    )

    # 输出优化后上报电量至csv
    df_opt_powers = pd.DataFrame(opt_powers, index=data['opt_result_index'], columns=['优化后上报电量'])
    df_opt_powers.to_csv(f'output/{station}/max_{poweropt.is_scenario}_scenario_{poweropt.is_chance_constr}_chance_{poweropt.is_robust}_rob_opt_powers.csv', encoding = 'utf-8-sig')

    # 输出优化后收益至csv
    df_reward = pd.DataFrame(reward)
    df_reward.index = pd.date_range(start=data['opt_result_index'][0].date(), periods=len(df_reward), freq='D')
    df_reward.to_csv(f'output/{station}/max_500_{poweropt.is_scenario}_scenario_{poweropt.is_chance_constr}_chance_{poweropt.is_robust}_rob_reward.csv', index=True, encoding = 'utf-8-sig')

    # 按月聚合
    # df_monthly = df_reward.resample('ME').agg({
    #             'daily_reward_dayahead': 'sum',
    #             'daily_reward_realtime': 'sum',
    #             'daily_cost_exam': 'sum',
    #             'daily_total_reward': 'sum',
    #             'daily_power_acc':'mean'
    # })
    # df_monthly['Month'] = df_monthly.index.strftime('%Y-%m')
    # df_monthly.set_index('Month', inplace=True)
    # df_monthly.to_csv(f'output/{poweropt.is_scenario}_scenario_{poweropt.is_chance_constr}_chance_{poweropt.is_robust}_rob_reward_resample.csv', index=True, encoding = 'utf-8-sig')
