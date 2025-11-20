# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/08 17:40:00
@Des     : 报量策略
@Author  : hehong.shen
"""
import os
import yaml
import logging
import traceback
import numpy as np
from pyscipopt import Model, quicksum


logger = logging.getLogger(__name__)
PARAMS_ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__), 'saved_params')


class PowerOpt:
    def __init__(self):
        """
        保量策略类
        """
        self.tune_powers = None  # solver调整量
        self.max_power = None
        self.site_type = None
        self.is_join_market = None
        self.pred_powers = None
        self.pred_da_clear_ratio = None
        self.pred_spread = None
        self.pred_spread_upper = None
        self.pred_spread_lower = None
        self.longterm_powers = None
        self.longterm_prices = None
        self.is_user_defined = None
        self.adjust_max = None
        self.adjust_min = None
        # self.bet = None
        self.length = None
        self.error_tolerance = None
        self.params_path = PARAMS_ABSOLUTE_PATH
        self.params = self.load_params()
        self.is_robust = None
        self.is_chance_constr = None
        self.is_scenario = None
        self.scenarios = None
        self.scenarios_probs = None
    def load_params(self):
        """
        加载数据获取配置文件
        """
        try:
            with open(os.path.join(self.params_path, 'trade_params.yaml'), 'r', encoding='utf-8') as f:
                ps = yaml.load(f, Loader=yaml.FullLoader)
            return ps
        except Exception as e:
            logger.error('参数yaml文件不存在', e)
            raise FileNotFoundError

    def init_params(self,
                    max_power,
                    site_type,
                    is_join_market,
                    pred_powers,
                    pred_da_limit_ratio,
                    pred_da_prices,
                    pred_id_prices,
                    longterm_powers,
                    longterm_prices,
                    is_user_defined,
                    adjust_max,
                    adjust_min,
                    is_robust,
                    pred_spread,
                    pred_spread_upper,
                    pred_spread_lower,
                    is_chance_constr,
                    is_scenario,
                    scenarios,
                    scenarios_probs):
        self.max_power = max_power
        self.site_type = site_type
        self.is_join_market = is_join_market
        self.pred_powers = pred_powers
        self.pred_da_clear_ratio = 1 - pred_da_limit_ratio  # 日前出清率
        # self.pred_spread = pred_id_prices - pred_da_prices  # 实时电价-日前电价
        self.pred_spread = pred_spread
        self.pred_spread_upper = pred_spread_upper
        self.pred_spread_lower = pred_spread_lower
        self.longterm_powers = longterm_powers
        self.longterm_prices = longterm_prices
        self.is_user_defined = is_user_defined
        self.adjust_max = adjust_max
        self.adjust_min = adjust_min
        self.is_robust = is_robust
        self.is_chance_constr = is_chance_constr
        self.is_scenario = is_scenario
        self.scenarios = scenarios
        self.scenarios_probs = scenarios_probs
        self.length = len(pred_powers)  # 输入数据长度
        self.error_tolerance = self.params['ERROR_TOLERANCE'][self.site_type]
        # if len(bet) == 0:
        #     self.bet = self.params['CAPITAL_POWER_RATIO'] * self.pred_powers
        #     self.bet[self.pred_spread > 0] *= -1
        # else:
        #     self.bet = bet
        # 结算电价平均处理
        # pml = self.params['PRICE_MEAN_LENGTH']
        # if pml > 1 and not self.length % self.params['PRICE_MEAN_LENGTH']:
        #     self.pred_spread = np.array([np.mean(self.pred_spread[int(i / pml) * pml:int(i / pml) * pml + pml])
        #                                  for i in range(self.length)])
        #     logger.info(f'结算电价按{self.params["PRICE_MEAN_LENGTH"]}个点取平均')

    def optimize(self,
                 max_power,
                 site_type,
                 is_join_market,
                 pred_powers,
                 pred_da_limit_ratio,
                 pred_da_prices,
                 pred_id_prices,
                 longterm_powers,
                 longterm_prices,
                 is_user_defined,
                 adjust_max,
                 adjust_min,
                 is_robust,
                 pred_spread,
                 pred_spread_upper,
                 pred_spread_lower,
                 is_chance_constr,
                 is_scenario,
                 scenarios,
                 scenarios_probs):
        """
        Parameters
        -----------
        max_power : float
            场站容量(MW)
        site_type : str
            场站类型
        is_join_market : int
            是否参与市场
        pred_powers : ndarray
            短期预测功率(MW)
        pred_da_limit_ratio : ndarray
            日前限电率
        pred_da_prices : ndarray
            日前预测价格
        pred_id_prices : DataFrame
            实时预测价格
        longterm_powers : ndarray
            中长期功率曲线
        longterm_prices : ndarray
            中长期电价曲线
        is_user_defined : binary
            是否自定义AI策略 1-是，0-否
        adjust_max : ndarray
            最大调整量，MW
        adjust_min : ndarray
            最小调整量，MW
        is_robust : binary
            是否鲁棒优化， 1-是，0-否
        pred_spread: ndarray
            50%分位数预测价差
        pred_spread_upper: ndarray
            上分位数预测价差
        pred_spread_lower: ndarray
            下分位数预测价差
        is_chance_constr: binary
            是否添加机会约束， 1-是，0-否
        is_scenario: binary
            是否基于场景优化， 1-是，0-否
        scenarios: ndarray
            场景集合
        scenarios_probs: ndarray
            各场景概率

        Returns
        -----------
        opt_powers : ndarray
            保量曲线，MW
        """
        # 初始化
        self.init_params(max_power,
                         site_type,
                         is_join_market,
                         pred_powers,
                         pred_da_limit_ratio,
                         pred_da_prices,
                         pred_id_prices,
                         longterm_powers,
                         longterm_prices,
                         is_user_defined,
                         adjust_max,
                         adjust_min,
                         is_robust,
                         pred_spread,
                        pred_spread_upper,
                        pred_spread_lower,
                         is_chance_constr,
                         is_scenario,
                         scenarios,
                         scenarios_probs)

        if np.sum(self.pred_spread == 0) == self.length:
            logger.info('无价差，solver调整量为0')
            self.tune_powers = np.zeros(self.length)
        else:
            # noinspection PyBroadException
            try:
                self.solver()
            except Exception:
                logger.warning('优化报错，不调', traceback.format_exc())
                self.tune_powers = np.zeros(self.length)
        opt_powers = self.tune_powers + self.pred_powers

        # 过滤功率很小且预测正价差的点
        filter_cap = self.params['FILTER_CAP'] if self.site_type == 'wind' else 0
        opt_powers[(opt_powers < max(filter_cap, self.max_power * 0.02)) & (self.pred_spread >= 0)] = 0

        # 负价差且不到中长期的点拉保底比例的装机或到中长期
        # if self.site_type == 'wind':
        #     idx = (opt_powers < self.longterm_powers) & (self.pred_spread <= 0)
        #     opt_powers[idx] = np.minimum(opt_powers[idx] + self.params['BE_TATIO'] * self.max_power,
        #                                  self.longterm_powers[idx])

        # 自定义AI策略处理
        if self.is_user_defined:
            idx = (self.pred_spread > 0)
            opt_powers[idx] = np.minimum(opt_powers[idx], self.pred_powers[idx] - self.adjust_min[idx])
            opt_powers[idx] = np.maximum(opt_powers[idx], self.pred_powers[idx] - self.adjust_max[idx])
            idx = (self.pred_spread < 0)
            opt_powers[idx] = np.maximum(opt_powers[idx], self.pred_powers[idx] + self.adjust_min[idx])
            opt_powers[idx] = np.minimum(opt_powers[idx], self.pred_powers[idx] + self.adjust_max[idx])

        # 平滑曲线
        opt_powers = self.smooth_all(opt_powers, smooth_times=self.params['SMOOTH_LEVEL'])

        logger.info(f'原始预测电量为{round(np.sum(self.pred_powers) / 4, 3)}，最终上报电量为{round(np.sum(opt_powers) / 4, 3)}')
        return opt_powers

    def solver(self):
        """
        优化求解器
        """
        solar_idx = self.get_idxs(self.params['SOLAR_TIMES'])
        clear_rate = self.pred_da_clear_ratio if self.is_join_market else \
            (1 - self.params['BASE_RATIO']) * self.pred_da_clear_ratio + self.params['BASE_RATIO']  # 由是否参与市场决定
        m = Model()
        # powers为决策变量，即曲线调整的量
        powers, powers_abs, deviation1, deviation2, x_plus, x_neg= {}, {}, {}, {}, {}, {}
        for i in range(self.length):
            powers[i] = m.addVar(vtype='C', lb=-self.pred_powers[i], ub=self.max_power - self.pred_powers[i])  # 调整量
            powers_abs[i] = m.addVar(vtype='C', lb=0, ub=self.max_power)  # 调整量的绝对值
            deviation1[i] = m.addVar(vtype='C', lb=0, ub=self.max_power)  # 日前偏差考核量(上报>1.2实发)
            deviation2[i] = m.addVar(vtype='C', lb=0, ub=self.max_power)  # 日前偏差考核量(0.8实发>上报)

            m.addCons(deviation1[i] >= powers[i] - self.error_tolerance * self.pred_powers[i])  # 偏差考核
            m.addCons(deviation2[i] >= -powers[i] - self.error_tolerance * self.pred_powers[i])
            # m.addCons(
            #     powers[i] >= min(-self.max_power * self.params['DOWN_LIMIT_RATIO'], 0))  # 下调幅度约束
            # m.addCons(powers[i] <= max(self.max_power * self.params['UP_LIMIT_RATIO'], 0))  # 上调幅度约束
            # 调整量不大于功率预测*20%
            m.addCons(powers_abs[i] <= self.params['UP_LIMIT_RATIO'] * self.pred_powers[i])
            # 调整量绝对值
            m.addCons(powers_abs[i] >= powers[i])
            m.addCons(powers_abs[i] >= -powers[i])

            if self.site_type == 'solar' and i not in solar_idx:  # 光伏无效出力时段不调
                m.addCons(powers[i] == 0)
            if self.is_robust:
                x_plus[i] = m.addVar(vtype='C', lb=0, ub=self.max_power)
                x_neg[i] = m.addVar(vtype='C', lb=0, ub=self.max_power)
                m.addCons(x_plus[i] >= powers[i])
                m.addCons(x_neg[i] >= -powers[i])
                m.addCons(x_plus[i] >= 0)
                m.addCons(x_neg[i] >= 0)
            if self.is_chance_constr:
                m.addCons(- powers[i] * self.pred_spread_lower[i] >= 0)
                m.addCons(- powers[i] * self.pred_spread_upper[i] >= 0)
        # obj
        if self.is_robust:
            robust_term = quicksum(
                (self.pred_spread_upper[i] * x_plus[i] + self.pred_spread_lower[i] * x_neg[i])
                * clear_rate for i in range(self.length)
            )

            m.setObjective(
                - robust_term
                - quicksum(deviation1[i] + deviation2[i] for i in range(self.length))
                * self.params['ASSESS_RATIO'] * self.params['ASSESS_PRICE']
                - quicksum(powers_abs[i] for i in range(self.length)) * 0.01,
                'maximize'
            )

        elif self.is_scenario:
            m.setObjective(
                sum(
                    sum(-self.scenarios[i, s] * clear_rate * powers[i] * self.scenarios_probs[s,i]  for i in range(self.length))
                    for s in range(len(self.scenarios_probs))
                )
                - sum(deviation1[i] + deviation2[i] for i in range(self.length))
                * self.params['ASSESS_RATIO'] * self.params['ASSESS_PRICE']
                - sum(powers_abs[i] for i in range(self.length)) * 0.01,
                "maximize")
            # m.setObjective(
            #     sum(
            #         self.scenarios_probs[s] *
            #         sum(-self.scenarios[i, s] * clear_rate * powers[i] for i in range(self.length))
            #         for s in range(len(self.scenarios_probs))
            #     )
            #     - sum(deviation1[i] + deviation2[i] for i in range(self.length))
            #     * self.params['ASSESS_RATIO'] * self.params['ASSESS_PRICE']
            #     - sum(powers_abs[i] for i in range(self.length)) * 0.01,
            #     "maximize"
            # )
        else:
            m.setObjective(np.dot(-self.pred_spread * clear_rate, [powers[i] for i in range(self.length)])
                               - quicksum([deviation1[i] + deviation2[i] for i in range(self.length)])
                               * self.params['ASSESS_RATIO'] * self.params['ASSESS_PRICE']
                               - quicksum(powers_abs[i] for i in range(self.length)) * 0.01, 'maximize')

        m.hideOutput()
        m.optimize()
        logger.info(f'优化求解状态为{str(m.getStatus())}')

        if m.getStatus() == 'optimal':
            self.tune_powers = np.array([m.getVal(powers[i]) for i in range(self.length)])
            self.dev1 = np.array([m.getVal(deviation1[i]) for i in range(self.length)])
            self.dev2 = np.array([m.getVal(deviation2[i]) for i in range(self.length)])
        else:
            self.tune_powers = np.zeros(self.length)  # 求解失败，不调
            self.dev1 = np.zeros(self.length)
            self.dev2 = np.zeros(self.length)
    def smooth_all(self, a, window_size=3, smooth_times=2):
        """
        曲线平滑

        Parameters
        ----------
        a : ndarray
            原始数据
        window_size : int
            滑动平均窗口大小
        smooth_times : int
            平滑次数

        Returns
        -----------
        a : ndarray
            时间idx
        """
        for i in range(smooth_times):
            out0 = np.convolve(a, np.ones(window_size, dtype=int), 'valid') / window_size
            r = np.arange(1, window_size - 1, 2)
            start = np.cumsum(a[:window_size - 1])[::2] / r
            stop = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
            a = np.concatenate((start, out0, stop))
        a[a >= self.max_power] = self.max_power
        a[a < 0.001] = 0
        return a

    @staticmethod
    def get_idxs(time_range):
        """
        获取idx

        Parameters
        -----------
        time_range : 1d-list
            时间范围

        Returns
        -----------
        idxs : list
            时间idx
        """

        def get_idx(times):
            temp = times.split(':')
            return int(temp[0]) * 4 + int(temp[1]) // 15

        idxs = [i for i in range(get_idx(time_range[0]), get_idx(time_range[1]) + 1)]

        return idxs