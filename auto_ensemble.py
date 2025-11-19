import numpy as np
import pandas as pd
import cvxpy as cp
from itertools import product
from etide.model_evaluation import price_accuracy_smape
from loguru import logger
from etide.utils import get_date
import os

_cur_file = os.path.dirname(__file__)


class AutoModelEnsemble:
    """
    模型自动化融合类

    :param model_list: list, 子模型列名
    :param target: str, 标签名
    :param params_grid: dict, 超参数网格
    :param opt_weights: list, 子模型初始权重配比
    :param hype_params_renew_freq: int, 超参更新频率（天）
    :param metric_func: function, 评估指标函数
    :param max_obj: bool, 评估指标是否越大越好
    """

    def __init__(self,
                 model_list,
                 target,
                 params_grid,
                 opt_weights,
                 hype_params_renew_freq,
                 metric_func=price_accuracy_smape,
                 max_obj=True
                 ):
        self.model_list = model_list
        self.target = target
        keys = params_grid.keys()
        self.params_grid = [dict(zip(keys, v)) for v in product(*params_grid.values())]
        self.metric_func = metric_func
        self.max_obj = max_obj
        self.opt_weights = np.array(opt_weights)
        self.weights_dist = None
        self.hist_test_scores = None
        self.hype_params_renew_freq = hype_params_renew_freq

    @staticmethod
    def _optimize(x_hist,
                  y_hist,
                  last_opt_weights,
                  decay
                  ):
        """
        最优权重求解模块

        :param x_hist: numpy.ndarray, 历史子模型预测值
        :param y_hist: numpy.ndarray, 历史标签真实值
        :param last_opt_weights: numpy.ndarray, 最新子模型权重配置
        :param decay: int, 超参数, 权重平滑系数
        :returns: opt_weights, 最优权重
        """
        n_models = x_hist.shape[1]
        weights_var = cp.Variable(n_models, nonneg=True)
        error = cp.Variable(x_hist.shape[0])
        constraints = [
            cp.sum(weights_var) == 1,
            error >= y_hist - x_hist @ weights_var,
            error >= x_hist @ weights_var - y_hist
        ]
        objective = cp.Minimize(cp.mean(error))
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        if problem.status == 'optimal':
            opt_weights = weights_var.value.round(4)
            if last_opt_weights is not None:
                opt_weights = decay * last_opt_weights + (1 - decay) * opt_weights
        else:
            logger.error(f"求解失败, 状态: {problem.status}, 使用算术平均权重")
            opt_weights = np.ones(n_models) / n_models
        return opt_weights

    def predict(self,
                data,
                pred_dates,
                threshold,
                window_len_range,
                decay,
                opt_weights
                ):
        """
        预测模块, 基于给定超参组合及权重信息, 引入权重更新机制, 并对待预测日期进行预测。

        :param data: pd.DataFrame, 输入历史数据
        :param pred_dates: pd.data_range(...), 待预测日期范围
        :param threshold: int, 超参, 权重更新紧迫度分数阈值
        :param window_len_range: int, 超参, 权重计算窗口长度
        :param decay: float, 超参, 权重计算目标函数中的衰减系数
        :param opt_weights: list, 子模型权重配比
        :returns: 融合预测值y_preds, 更新后权重opt_weights
        """
        y_preds = []
        for pred_date in pred_dates:
            pred_date = get_date(pred_date)
            # 判断历史精度变化
            if self.hist_test_scores is None:
                if opt_weights is not None:
                    pass
                else:
                    logger.info(f"初始权重为空, 且预测日前无历史数据, 使用算术平均权重")
                    opt_weights = np.ones(len(self.model_list)) / len(self.model_list)
            else:
                # 判断精度绝对下降值是否超过阈值
                thres_st = get_date(pred_date, shift_days=-2)
                thres_et = get_date(pred_date, shift_days=-1)
                delta_score = self.hist_test_scores.loc[thres_et].values - self.hist_test_scores.loc[thres_st].values
                if self.max_obj:
                    precision_score = - delta_score
                else:
                    precision_score = delta_score
                weights_score = self.weights_dist
                if weights_score is not None:
                    urgency_score = precision_score + weights_score
                else:
                    urgency_score = precision_score
                if urgency_score >= threshold:
                    # 计算自适应窗口长度
                    window_len = max(1, min(int(window_len_range * 10 * urgency_score), window_len_range))
                    # 取出自适应窗口历史数据
                    end_date = get_date(pred_date, shift_days=-1)
                    start_date = get_date(end_date, shift_days=-window_len)
                    window_df = data.loc[start_date:end_date].dropna()
                    # 在窗口内求解最优权重
                    x_hist = window_df[self.model_list].values
                    y_hist = window_df[self.target].values
                    opt_weights = self._optimize(x_hist, y_hist, opt_weights, decay)
                else:
                    pass
            x_day = data.loc[pred_date, self.model_list].values
            y_pred_day = x_day @ opt_weights
            y_preds.append(y_pred_day)
        y_preds = np.concatenate(y_preds)
        return y_preds, opt_weights

    def train(self,
              data,
              train_start_date,
              train_end_date,
              opt_weights
              ):
        """
        超参寻优模块：基于给定超参寻优网格, 根据精度寻找最优超参组合。
        当前版本包含3个超参数, 分别是融合模型权重更新紧迫度阈值threshold、权重平滑系数decay以及滑动窗口最大长度window_len_range

        :param data: pd.DataFrame, 历史子模型预测及标签值
        :param train_start_date: str, %Y-%m-%d, 训练起始时间
        :param train_end_date: str, %Y-%m-%d, 训练结束时间
        :param opt_weights: list, 子模型权重配比
        :returns: 最优超参best_threshold, best_decay, best_window_len_range
        """
        train_dates = pd.date_range(train_start_date, train_end_date, freq='D').to_period('D')
        if self.max_obj:
            best_score = -np.inf
        else:
            best_score = np.inf
        best_params = None

        for params in self.params_grid:
            y_params_pred, _ = self.predict(data,
                                            train_dates,
                                            params['threshold'],
                                            params['window_len_range'],
                                            params['decay'],
                                            opt_weights
                                            )
            y_true = data.loc[train_start_date:train_end_date, self.target].values
            final_score = self.metric_func(y_true, y_params_pred)
            if self.max_obj:
                if final_score > best_score:
                    best_score = final_score
                    best_params = (params['threshold'], params['window_len_range'], params['decay'])
            else:
                if final_score < best_score:
                    best_score = final_score
                    best_params = (params['threshold'], params['window_len_range'], params['decay'])

        best_threshold = best_params[0]
        best_window_len_range = best_params[1]
        best_decay = best_params[2]

        return best_threshold, best_decay, best_window_len_range

    def run(self,
            data,
            test_start_date,
            test_end_date):
        """
        用于在历史数据集上搜索最优超参更新频率及最优超参。

        :param data: pd.DataFrame, 历史子模型预测及标签值
        :param test_start_date: str, 回测起始日期
        :param test_end_date: str, 回测结束日期
        :returns: 回测区间预测值及精度backtest_df, 最终子模型权重配置final_opt_weights
        """
        test_dates = pd.date_range(test_start_date, test_end_date, freq='D').tolist()

        # 日期范围校验
        if test_end_date > get_date(data.index[-1]) or test_start_date < get_date(data.index[0]):
            raise KeyError(
                f'输入回测日期范围{test_start_date}~{test_end_date}超出历史数据日期范围{get_date(data.index[0])}~{get_date(data.index[-1])}')

        # 计算初始化历史test_scores
        df_hist = data.loc[data.index < str(test_start_date)]
        hist_dates = df_hist.index.strftime('%Y-%m-%d').unique().tolist()
        hist_test_scores = []
        if len(hist_dates) >= 2:
            for current_date in hist_dates:
                x_hist_day = df_hist.loc[current_date, self.model_list].values
                y_hist_day = df_hist.loc[current_date, self.target].values
                y_pred_day = x_hist_day @ self.opt_weights
                day_score = self.metric_func(y_hist_day, y_pred_day)
                hist_test_scores.append(day_score)
        else:
            hist_test_scores = None
        hist_scores_df = pd.DataFrame(data=hist_test_scores, index=get_date(df_hist.index).unique().tolist())
        self.hist_test_scores = hist_scores_df

        # 对于待回测日期进行回测
        prediction = []
        best_threshold, best_decay, best_window_len_range = None, None, None
        for i, current_date in enumerate(test_dates):
            current_date = get_date(current_date)
            if data.loc[current_date:current_date, self.model_list].isna().all(axis=None):
                continue
            # 定期调用train更新超参数
            if i % self.hype_params_renew_freq == 0:
                train_end_date = get_date(current_date, shift_days=-1)
                train_start_date = get_date(train_end_date, shift_days=-self.hype_params_renew_freq)
                if data.loc[train_start_date:train_end_date, self.model_list].isna().all(axis=None):
                    continue
                train_init_opt_weights = self.opt_weights
                best_threshold, best_decay, best_window_len_range = self.train(data,
                                                                               train_start_date,
                                                                               train_end_date,
                                                                               train_init_opt_weights)
                logger.info(
                    f'{current_date}更新超参, best_threshold, best_decay, best_window_len_range为{best_threshold, best_decay, best_window_len_range}')
            day_pred, renew_opt_weights = self.predict(data,
                                                       [str(current_date)],
                                                       best_threshold,
                                                       best_window_len_range,
                                                       best_decay,
                                                       self.opt_weights
                                                       )
            # 计算权重变化（欧式距离）
            self.weights_dist = np.sqrt(np.sum((renew_opt_weights - self.opt_weights) ** 2))
            # 更新全局权重
            self.opt_weights = renew_opt_weights
            prediction.append(day_pred)
            # 实时更新历史精度
            val = self.metric_func(data.loc[current_date, self.target].values, day_pred)
            new_row = pd.DataFrame(data=[[val]],
                                   index=[str(current_date)],
                                   columns=[0]
                                   )
            self.hist_test_scores = pd.concat([self.hist_test_scores, new_row])
        prediction = np.concatenate(prediction)

        # 计算整体平均分数
        y_hist = data.loc[test_start_date:test_end_date, self.target].values
        test_scores = self.metric_func(y_hist, prediction)
        logger.info(
            f'回测完成,在{self.hype_params_renew_freq}天的超参更新频率下,{test_start_date}~{test_end_date}的{self.metric_func.__name__}为{test_scores},最终权重配置为{self.opt_weights}')
        backtest_df = pd.DataFrame(prediction, index=pd.date_range(start=test_start_date, periods=len(prediction), freq='15T'),
                            columns=['融合预测'])
        final_opt_weights = list(self.opt_weights)

        return backtest_df, final_opt_weights
