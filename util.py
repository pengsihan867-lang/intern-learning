import numpy as np
import pandas as pd
import os
import yaml
from sklearn.cluster import KMeans
from scipy.stats import qmc
from scipy.interpolate import interp1d
from scipy.stats import laplace
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist,pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

cur_dir = os.path.dirname(os.path.abspath(__file__))

# 准备优化及收益计算的输入数据
def data_prepare(start_date, end_date, station):
    # 参数读取
    params_dir = os.path.join(cur_dir, 'saved_params', 'trade_params.yaml')
    with open(params_dir, 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)
    # 全分位数预测价差数据准备
    df_full_quantiles =  pd.read_csv('data/full_quantiles_pred_price_data_0101_0731.csv',index_col=0, parse_dates=True)
    df_full_quantiles = df_full_quantiles.loc[start_date:end_date]

    # 电价与功率原始数据准备
    df_price = pd.read_csv('data/price_data.csv',index_col=0, parse_dates=True)
    df_price = df_price.loc[start_date:end_date]
    df_power = pd.read_excel(f'data/{station}_power_data.xlsx', index_col=2, parse_dates=True)
    df_power = df_power.loc[start_date:end_date]

    # 获取测试周期的日前与实时的电价，用于输入优化及计算收益
    pred_da_prices = df_price['日前电价_d+1预测'].values
    pred_id_prices = df_price['实时电价_d+1预测'].values
    dayahead_price = df_power['DA_CLEARING_PRICE'].values
    realtime_price = df_power['ID_CLEARING_PRICE'].values
    pred_spread = df_full_quantiles ['q0.50']
    pred_spread_upper = df_full_quantiles ['q0.95']
    pred_spread_lower = df_full_quantiles ['q0.05']

    # 获取测试周期的功率，用于输入优化及计算收益
    pred_powers = df_power['POWER_FORECAST'].values
    real_powers = df_power['RT_ONLINE_POWER'].values
    ai_dayahead_powers = df_power['DAYAHEAD_REPORT_POWER'].values
    data = {
        'params': params, # yaml参数
        'pred_da_prices': pred_da_prices,# 预测日前电价
        'pred_id_prices': pred_id_prices,# 预测实时电价
        'dayahead_price': dayahead_price,# 真实日前电价
        'realtime_price': realtime_price,# 真实实时电价
        'pred_powers': pred_powers,# 预测功率
        'real_powers': real_powers,# 实发功率
        'ai_dayahead_powers': ai_dayahead_powers,# AI申报功率
        'pred_spread': pred_spread, # 50%分位数预测价差
        'pred_spread_upper': pred_spread+1000, # 上分位数预测价差
        'pred_spread_lower': pred_spread-1000, # 下分位数预测价差
        'opt_result_index': df_price.index,
        'df_full_quantiles': df_full_quantiles # 全分位数预测价差
    }
    return data

# 功率准确率计算函数
def calc_rmse_accuracy(pred, real, cap):
    """
    计算RMSE准确率

    参数:
        pred (Series): 预测值
        real (Series): 实际值
        cap (float): 容量

    返回:
        float: RMSE准确率
    """
    return 1 - np.sqrt(np.nanmean(((real - pred) / cap) ** 2))

# 收益计算函数（按日）
def reward_calculate(
        dayahead_powers,  # 调整后日前申报量
        real_powers,  # 实发功率，真实值
        dayahead_price,  # 日前电价，真实值
        realtime_price,  # 实时电价，真实值
        assess_ratio,  # 偏差考核比例
        assess_price,  # 偏差考核基准价格
        error_tolerance,  # 新能源获利回收中偏差允许范围
        max_power , # 场站容量
        set_time_granularity = 15 # 市场结算颗粒度
):
    T0 = set_time_granularity / 60.0
    delta_powers = real_powers - dayahead_powers  # 实发功率-调整后日前申报功率
    result = []
    for start in range(0, len(dayahead_powers), 96):
        end = start + 96

        daily_power_acc = calc_rmse_accuracy(pred=dayahead_powers[range(start, end)], real=real_powers[range(start, end)], cap= max_power)
        daily_reward_dayahead = 0.0
        daily_reward_realtime = 0.0
        daily_cost_exam = 0.0

        # 每15min考核/收益计算
        for i in range(start, end):
            # 考核成本计算
            deviation1 = max((dayahead_powers[i] - (1 + error_tolerance) * real_powers[i]), 0)  # 正偏差量
            deviation2 = max(((1 - error_tolerance) * real_powers[i] - dayahead_powers[i]), 0)  # 负偏差量
            cost_exam = (deviation1 + deviation2) * assess_ratio * assess_price *T0   # 考核成本

            # 日前与实时收益计算
            reward_dayahead = dayahead_powers[i] * dayahead_price[i] *T0  # 日前收益
            reward_realtime = delta_powers[i] * realtime_price[i] *T0  # 实时收益

            # 计入当日
            daily_reward_dayahead += reward_dayahead
            daily_reward_realtime += reward_realtime
            daily_cost_exam += cost_exam

        # 当日净收益
        daily_total_reward = daily_reward_dayahead + daily_reward_realtime - daily_cost_exam
        result.append({
            'daily_reward_dayahead': daily_reward_dayahead,
            'daily_reward_realtime': daily_reward_realtime,
            'daily_cost_exam': daily_cost_exam,
            'daily_total_reward': daily_total_reward,
            'daily_power_acc': daily_power_acc
        })
    return result

def inverse_cdf(df_full_quantiles):
    # 提取分位数水平
    q_levels = np.array([float(col[1:]) for col in df_full_quantiles.columns])

    # 为每个时间步构建逆 CDF（即分位数函数）
    inverse_cdfs = []
    for _, row in df_full_quantiles.iterrows():
        q_orig = row.values.astype(float)
        y_min, y_max = q_orig.min(), q_orig.max()
        num_grid = max(100, len(q_levels) * 2)
        y_grid = np.linspace(y_min, y_max, num_grid)

        # 构造经验 CDF: F(y) = ∫ 1{q(τ) ≤ y} dτ over τ in q_levels
        F_y = np.zeros_like(y_grid)
        for i, y in enumerate(y_grid):
            indicator = (q_orig <= y).astype(float)  # 1{q(τ) ≤ y}
            F_y[i] = np.trapz(indicator, x=q_levels)  # 梯形法则积分
        # 确保非减（数值稳定性）
        if not np.all(np.diff(F_y) >= 0):
            F_y = np.maximum.accumulate(F_y)

        # 找有效区间（避开 F_y=0 和 F_y=1 的平台）
        valid_idx = (F_y > 0) & (F_y < 1)

        if valid_idx.sum() < 10:
            # 回退：直接排序插值（防止插值失败）
            sorted_q = np.sort(q_orig)
            interp_func = interp1d(
                q_levels, sorted_q,
                kind='linear', bounds_error=False,
                fill_value=(sorted_q[0], sorted_q[-1])
            )
        else:
            try:
                interp_func = interp1d(
                    F_y[valid_idx], y_grid[valid_idx],
                    kind='linear', bounds_error=False,
                    fill_value=(y_grid[0], y_grid[-1])
                )
            except:
                # 再次失败则回退
                sorted_q = np.sort(q_orig)
                interp_func = interp1d(
                    q_levels, sorted_q,
                    kind='linear', bounds_error=False,
                    fill_value=(sorted_q[0], sorted_q[-1])
                )
        inverse_cdfs.append(interp_func)
    return inverse_cdfs

def generate_and_reduce_quantile_scenarios(
    df_full_quantiles,
    pred_da_prices,
    n_scenarios,
    n_reduced,
    set_time_granularity=15,
    seed=50
):
    """
    将分位数预测 DataFrame 转换为多个随机场景，并按天进行场景削减。
    返回削减后的场景和对应概率，结构与 generate_laplace_scenarios 一致。

    Parameters:
    -----------
    df_full_quantiles : pd.DataFrame
        列名为 q0.01, q0.02, ..., q0.99 的分位数预测结果 (nT, nq)
    pred_da_prices : array-like
        预测的日前电价序列 (nT,)
    n_scenarios : int
        每天生成的初始场景数量
    n_reduced : int
        每天削减后的场景数量
    set_time_granularity : int
        时间粒度（分钟），默认 15 分钟
    seed : int
        随机种子

    Returns:
    --------
    reduced_scenarios : np.ndarray
        形状为 (nT, n_reduced)，reduced_scenarios[t, k] 表示第 t 个时刻第 k 个场景的值
    reduced_probs : np.ndarray
        形状为 (n_reduced, nT)，reduced_probs[k, t] 表示第 t 天第 k 个场景的概率
    """
    # 转换输入
    pred_da_prices = np.asarray(pred_da_prices)
    nT = len(pred_da_prices)

    # 计算每天的时间点数和总天数
    n_times = int(24 * 60 / set_time_granularity)  # 如 96 (15分钟粒度)
    total_days = nT // n_times
    assert nT == total_days * n_times, f"总时间点 {nT} 不能被每天 {n_times} 整除"

    # 拟合cdf
    inverse_cdfs = inverse_cdf(df_full_quantiles)

    # 初始化输出
    reduced_scenarios = np.zeros((nT, n_reduced))  # (nT, n_reduced)
    reduced_probs = np.zeros((n_reduced, nT))       # (n_reduced, nT)
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n_scenarios).flatten()
    # 按天处理
    for day in range(total_days):
        print(f"Processing day {day + 1}/{total_days}")
        start_t = day * n_times
        end_t = start_t + n_times

        day_scenarios = np.zeros((n_scenarios, n_times))
        for t_idx in range(n_times):
            global_t = start_t + t_idx
            day_scenarios[:, t_idx] = inverse_cdfs[global_t](u)
            if pred_da_prices[global_t] <= -80:
                day_scenarios[:, t_idx] = 10.0

        # 场景削减
        reduced_day, probs = reduce_scenarios_heuristic(n_reduced, day_scenarios)
        # reduced_day: (n_times, n_reduced), probs: (n_reduced,)

        # 填充结果
        reduced_scenarios[start_t:end_t, :] = reduced_day
        reduced_probs[:, start_t:end_t] = probs[:, np.newaxis]  # (n_reduced, n_times)

    return reduced_scenarios, reduced_probs

def generate_and_reduce_laplace_scenarios(
        n_scene,
        lower,
        upper,
        pred_da_price,
        n_reduced,
        set_time_granularity=15,
        confidence=0.90,
):
    """
    根据给定的置信区间下界和上界，拟合拉普拉斯分布的尺度参数 b，
    并使用拉丁超立方抽样（LHS）生成符合该分布的场景，同时施加价格边界约束。

    参数:
    - n_scene: int，生成的场景数量
    - lower: array-like，每个时间段的置信下界 (nT,)
    - upper: array-like，每个时间段的置信上界 (nT,)
    - da_price: array-like，每个时间段的日前市场价格 (nT,)
    - confidence: float，置信水平，默认 0.90（即 90% 区间）

    返回:
    - scenarios: 2D numpy array，形状为 (n_scene, nT)
    """

    lower = np.asarray(lower)
    upper = np.asarray(upper)

    pred_da_price = np.asarray(pred_da_price)
    nT = lower.size
    n_times = int(24*60/set_time_granularity)
    total_days = int(nT/n_times)
    assert upper.size == nT and pred_da_price.size == nT, "lower, upper, da_price 长度必须一致"

    # 计算拉普拉斯分布的中心 mu
    mu = (lower + upper) / 2

    # 求解每个时间段的尺度参数 b
    b_values = np.full(nT, np.nan)

    def error_func(b, mu_val, lower_bound, conf):
        return laplace.ppf((1 - conf) / 2, loc=mu_val, scale=b) - lower_bound

    for i in range(nT):
        try:
            res = root_scalar(
                error_func,
                args=(mu[i], lower[i], confidence),
                bracket=[1e-6, 1000],
                method='brentq'
            )
            b_values[i] = res.root
        except ValueError:
            pass  # 保持 np.nan

    # 生成场景
    scenarios = np.zeros((n_scene, nT))
    reduced_scenarios = np.zeros((nT, n_reduced))
    reduced_probs = np.zeros((n_reduced, nT))

    for i in range(nT):
        if np.isnan(b_values[i]):
            # 求解失败时，使用均值代替（退化为确定性值）
            scenarios[:, i] = mu[i]
        else:
            # LHS 抽样 + 拉普拉斯逆变换
            sampler = qmc.LatinHypercube(d=1, seed=50)
            u = sampler.random(n_scene).flatten()
            scenarios[:, i] = laplace.ppf(q=u, loc=mu[i], scale=b_values[i])
            # 后处理：基于预测日前电价，校正价差范围
            if pred_da_price[i] <= -80:
                for s in range(n_scene):
                    scenarios[s, i] = 10

    for total_day in range(total_days):
        # 提取当天的原始场景数据
        day_scenarios = scenarios[:, (total_day * n_times):(total_day * n_times + n_times)]

        # 场景削减
        reduced_day, probs = reduce_scenarios_heuristic(n_reduced, day_scenarios)

        # 填充削减后的场景 (n_times, n_reduced)
        start_idx = total_day * n_times
        end_idx = start_idx + n_times
        reduced_scenarios[start_idx:end_idx, :] = reduced_day  # shape: (n_times, n_reduced)
        reduced_probs[:, start_idx:end_idx] = probs[:, np.newaxis]  # shape: (n_reduced, n_times)
    return reduced_scenarios, reduced_probs

def reduce_scenarios_kmeans(n_reduced, scenarios):
    """
    使用K-means聚类进行场景削减
    :param n_reduced: 削减目标场景数 (int)
    """

    kmeans = KMeans(n_clusters=n_reduced, random_state=42).fit(scenarios)
    reduced_scenarios = kmeans.cluster_centers_
    unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    reduced_probs = np.zeros(n_reduced)  # 初始化概率数组
    reduced_probs[unique_labels] = counts / np.sum(counts)  # 填充实际计数的概率

    return reduced_scenarios.T, reduced_probs


def reduce_scenarios_ffs(n_reduced, scenarios):
    """
    快速前向选择（等概率场景）
    """
    n_scenarios, n_features = scenarios.shape

    # 距离矩阵
    dist_matrix = cdist(scenarios, scenarios, 'euclidean')

    # 初始选择：任选一个（比如第一个）
    selected_idx = [0]

    while len(selected_idx) < n_reduced:
        min_total_cost = np.inf
        best_candidate = -1

        for i in range(n_scenarios):
            if i in selected_idx:
                continue
            candidate_set = selected_idx + [i]
            cost = 0.0
            for j in range(n_scenarios):
                min_dist = dist_matrix[j, candidate_set].min()
                cost += min_dist  # 等概率，可忽略权重（或最后除以 n_scenarios）
            if cost < min_total_cost:
                min_total_cost = cost
                best_candidate = i
        selected_idx.append(best_candidate)

    reduced_scenarios = scenarios[selected_idx]

    # 计算每个原始场景最近的代表场景，用于分配概率
    dist_to_reduced = cdist(scenarios, reduced_scenarios, 'euclidean')
    closest_cluster = np.argmin(dist_to_reduced, axis=1)

    reduced_probs = np.bincount(closest_cluster, minlength=n_reduced) / n_scenarios

    return reduced_scenarios.T, reduced_probs

def reduce_scenarios_hierarchical(n_reduced, scenarios):
    n_scenarios = scenarios.shape[0]
    if n_reduced >= n_scenarios:
        return scenarios.T, np.ones(n_scenarios) / n_scenarios

    # 层次聚类
    link_mat = linkage(scenarios, method='ward')  # 仅适用于欧氏距离
    cluster_labels = fcluster(link_mat, n_reduced, criterion='maxclust') - 1  # 0-based

    reduced_scenarios = []
    reduced_probs = []

    for k in range(n_reduced):
        mask = (cluster_labels == k)
        if np.any(mask):
            # 使用簇内均值作为代表场景
            center = scenarios[mask].mean(axis=0)
            prob = mask.sum() / n_scenarios  # 等概率下：数量占比即概率
            reduced_scenarios.append(center)
            reduced_probs.append(prob)

    return np.array(reduced_scenarios).T, np.array(reduced_probs)

def reduce_scenarios_heuristic(n_reduced, scenarios, threshold=1e-4):
    """
    基于启发式同步回带策略的场景削减（正确处理概率累加）

    Parameters:
    - scenarios: np.array, shape (N, T)
    - n_reduced: int, 目标场景数
    - threshold: float, 删除影响容忍度（用于启发式判断）

    Returns:
    - reduced_scenarios: np.array, shape (n_reduced, T)
    - probabilities: np.array, shape (n_reduced,), 非等概率
    """
    N, T = scenarios.shape

    if n_reduced >= N:
        prob = np.full(N, 1.0 / N)
        return scenarios.copy(), prob

    # 当前保留的场景索引（在原始场景中的位置）
    current_indices = np.arange(N)
    # 初始每个场景概率为 1/N
    probs = np.full(N, 1.0 / N)

    # 距离矩阵缓存函数
    def get_distance_matrix(scens):
        return squareform(pdist(scens))

    while len(current_indices) > n_reduced:
        current_scens = scenarios[current_indices]
        n_curr = len(current_scens)

        # 步骤1: 计算当前保留场景间的距离矩阵
        dist_matrix = get_distance_matrix(current_scens)

        # 步骤2: 启发式评分 —— 平均距离最小的场景最“冗余”
        avg_distances = np.mean(dist_matrix + np.eye(n_curr) * 1e9, axis=1)  # 排除自身
        candidate_idx_in_current = np.argmin(avg_distances)  # 当前集合中的索引
        candidate_global_idx = current_indices[candidate_idx_in_current]  # 原始数据中的索引

        # 步骤3: 找到该场景的最近邻（最相似的保留场景）
        dist_to_others = dist_matrix[candidate_idx_in_current]
        dist_to_others[candidate_idx_in_current] = np.inf  # 排除自己
        nearest_idx_in_current = np.argmin(dist_to_others)
        nearest_global_idx = current_indices[nearest_idx_in_current]

        # 步骤4: 评估删除影响（可基于距离变化、概率转移量等）
        # 这里我们用“删除场景的概率大小”作为影响代理（越大影响越大）
        impact = probs[candidate_global_idx]  # 概率越大，影响越大

        # 判断是否接受删除（这里 threshold 可理解为最大允许单次删除概率）
        if impact < threshold:
            # 接受删除：将该场景的概率转移给最近邻
            probs[nearest_global_idx] += probs[candidate_global_idx]
            # 从当前索引中移除
            current_indices = np.delete(current_indices, candidate_idx_in_current)
        else:
            # 不接受删除：尝试下一个最冗余的（简单策略：跳过，继续）
            # 实际中可提高阈值或调整启发式
            avg_distances[candidate_idx_in_current] = np.inf
            next_candidate = np.argmin(avg_distances)
            next_global_idx = current_indices[next_candidate]

            # 找最近邻
            dist_row = dist_matrix[next_candidate]
            dist_row[next_candidate] = np.inf
            nearest_other = np.argmin(dist_row)
            nearest_other_global = current_indices[nearest_other]

            # 转移概率并删除
            probs[nearest_other_global] += probs[next_global_idx]
            current_indices = np.delete(current_indices, next_candidate)
        # print(f"当前剩余场景数: {len(current_indices)}\n")
    # 提取最终场景和概率
    final_scenarios = scenarios[current_indices]
    final_probabilities = probs[current_indices]

    # 归一化（理论上应守恒，但防止浮点误差）
    final_probabilities /= final_probabilities.sum()

    return final_scenarios.T, final_probabilities