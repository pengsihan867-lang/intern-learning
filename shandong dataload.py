from datetime import datetime, timedelta

import pandas as pd
from etide.trading_data_access import DataLoaderEDP
from etide.utils import get_date  # etide.utils 是模块，不是子包，需从这里导入


class ShandongMoreData(DataLoaderEDP):

    # 获取数据
    def _get_data(self, start_date, end_date, data_type):
        startDay = start_date
        endDay = get_date(end_date, shift_days=1)
        url = (
            f"https://{self.daas_domain_name}/trading-terminal-daas-service/v1/center-base/province-actual-public-info"
            f"?provinceCode=37&startDay={startDay}&endDay={endDay}&dataType={data_type}"
        )
        return self.get_daas_data(url, if_change_time=False)

    # 将 daily 数据展开为 96 个 15min 点
    @staticmethod
    def convert_daily_to_15min(df_daily):
        if df_daily.empty:
            return pd.DataFrame(columns=df_daily.columns)

        if not isinstance(df_daily.index, pd.DatetimeIndex):
            df_daily.index = pd.to_datetime(df_daily.index)

        start_date = df_daily.index.min()
        end_date = df_daily.index.max() + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

        fifteen_min_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq='15min'
        )
        df_15min = pd.DataFrame(index=fifteen_min_index, columns=df_daily.columns)

        for date in df_daily.index:
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_15min_points = pd.date_range(
                start=day_start,
                periods=96,
                freq='15min'
            )
            for col in df_daily.columns:
                df_15min.loc[day_15min_points, col] = df_daily.loc[date, col]

        return df_15min

    # ======== 通用内部工具函数 ========
    @staticmethod
    def _expand_daily_list_to_15min(df_daily: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """content 是长度 96 的 list[float] 时，展开为 15min 序列。"""
        if df_daily.empty:
            return pd.DataFrame(columns=[col_name])

        dfs = []
        for day, row in df_daily.iterrows():
            values = row.get("content")
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                continue
            # 以当天 00:00 起，每 15 分钟一个点
            day_ts = pd.to_datetime(day)
            idx = pd.date_range(start=day_ts, periods=len(values), freq="15min")
            s = pd.Series(values, index=idx, name=col_name)
            dfs.append(s.to_frame())

        if not dfs:
            return pd.DataFrame(columns=[col_name])
        return pd.concat(dfs).sort_index()

    @staticmethod
    def _expand_daily_listdict_to_15min(df_daily: pd.DataFrame) -> pd.DataFrame:
        """content 是长度 96 的 list[dict] 时，展开为 15min 序列。"""
        if df_daily.empty:
            return pd.DataFrame()

        dfs = []
        for day, row in df_daily.iterrows():
            records = row.get("content")
            if not isinstance(records, (list, tuple)) or len(records) == 0:
                continue
            df_c = pd.DataFrame(records)
            day_ts = pd.to_datetime(day)
            idx = pd.date_range(start=day_ts, periods=len(df_c), freq="15min")
            df_c.index = idx
            dfs.append(df_c)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).sort_index()

    @staticmethod
    def _prepare_daily_content_df(raw) -> pd.DataFrame:
        """将原始返回整理为以 date 为索引、仅含 content 的 DataFrame。"""
        df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if df.empty or "time" not in df.columns or "content" not in df.columns:
            return pd.DataFrame(columns=["content"])
        return df.set_index("time")[["content"]]

    # ====================== 各种 dataType 封装 ======================

    # ---- 检修容量（日值 × 1）→ 15min 复制 ----
    def overhaulCapacity(self, start_date, end_date):
        """检修容量预测。"""
        raw = self._get_data(start_date, end_date, "overhaulCapacity")
        raw_df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if raw_df.empty or "time" not in raw_df.columns or "content" not in raw_df.columns:
            return pd.DataFrame(columns=["检修容量"])

        def extract_content(value):
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                first = value[0]
                if isinstance(first, (list, tuple)) and first:
                    return first[0]
                if isinstance(first, dict) and first:
                    # 取字典第一个值
                    for _, v in first.items():
                        return v
                return first
            if isinstance(value, dict) and value:
                for _, v in value.items():
                    return v
            return value

        df = raw_df.set_index("time")[["content"]]
        df["content"] = df["content"].apply(extract_content)
        df.index = pd.to_datetime(df.index)
        df.columns = ["检修容量"]

        return self.convert_daily_to_15min(df)

    # ---- 一类：content 为 list[float]，长度 96 ----

    def waterPowerWeekForecast(self, start_date, end_date):
        """水电含抽需。"""
        raw = self._get_data(start_date, end_date, "waterPowerWeekForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["水电含抽需"])
        return self._expand_daily_list_to_15min(df_daily, "水电含抽需")

    def newEnergyWeekForecast(self, start_date, end_date):
        """新能源。"""
        raw = self._get_data(start_date, end_date, "newEnergyWeekForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["新能源"])
        return self._expand_daily_list_to_15min(df_daily, "新能源")

    def loadRegulationWeekForecast(self, start_date, end_date):
        """直调负荷预测。"""
        raw = self._get_data(start_date, end_date, "loadRegulationWeekForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["直调负荷"])
        return self._expand_daily_list_to_15min(df_daily, "直调负荷")

    def totalPowerWeekForecast(self, start_date, end_date):
        """全网负荷预测。"""
        raw = self._get_data(start_date, end_date, "totalPowerWeekForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["全网负荷"])
        return self._expand_daily_list_to_15min(df_daily, "全网负荷")

    def loadRegulationReal(self, start_date, end_date):
        """实际负荷-直调负荷。"""
        raw = self._get_data(start_date, end_date, "loadRegulationReal")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["实际负荷-直调负荷"])
        return self._expand_daily_list_to_15min(df_daily, "实际负荷-直调负荷")

    def totalPowerReal(self, start_date, end_date):
        """实际负荷-全网负荷。"""
        raw = self._get_data(start_date, end_date, "totalPowerReal")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["实际负荷-全网负荷"])
        return self._expand_daily_list_to_15min(df_daily, "实际负荷-全网负荷")

    def totalPowerPreForecast(self, start_date, end_date):
        """日前预计划-全网负荷。"""
        raw = self._get_data(start_date, end_date, "totalPowerPreForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["日前预计划-全网负荷"])
        return self._expand_daily_list_to_15min(df_daily, "日前预计划-全网负荷")

    def loadRegulationPreForecast(self, start_date, end_date):
        """日前预计划-直调负荷。"""
        raw = self._get_data(start_date, end_date, "loadRegulationPreForecast")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame(columns=["日前预计划-直调负荷"])
        return self._expand_daily_list_to_15min(df_daily, "日前预计划-直调负荷")

    # ---- 二类：电源出清电量及台数（list[dict]，len=96） ----

    def dayAheadSourceClearElecUnits(self, start_date, end_date):
        """日前各类电源出清电量及台数。"""
        raw = self._get_data(start_date, end_date, "dayAheadSourceClearElecUnits")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame()
        return self._expand_daily_listdict_to_15min(df_daily)

    def realTimeSourceClearElecUnits(self, start_date, end_date):
        """实时各类电源出清电量及台数。"""
        raw = self._get_data(start_date, end_date, "realTimeSourceClearElecUnits")
        df_daily = self._prepare_daily_content_df(raw)
        if df_daily.empty:
            return pd.DataFrame()
        return self._expand_daily_listdict_to_15min(df_daily)

    # ---- 三类：日机组最大出力（list[dict{unitType, unitDataList}]） ----

    def dailyUnitMaximumPower(self, start_date, end_date):
        """日机组最大出力，展开到机组明细。"""
        raw = self._get_data(start_date, end_date, "dailyUnitMaximumPower")
        df_daily = pd.DataFrame(raw)
        if df_daily.empty:
            return pd.DataFrame()

        rows = []
        for _, row in df_daily.iterrows():
            day = pd.to_datetime(row.get("time"))
            content = row.get("content") or []
            for block in content:
                if not isinstance(block, dict):
                    continue
                unit_type = block.get("unitType")
                for u in block.get("unitDataList", []) or []:
                    rec = {"date": day, "unitType": unit_type}
                    if isinstance(u, dict):
                        rec.update(u)
                    rows.append(rec)
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        result.set_index("date", inplace=True)
        return result.sort_index()

    # ---- 四类：实时机组开停机状态（site/machine/status 嵌套） ----

    def realTimeMachineRunStopStatus(self, start_date, end_date):
        """实时各机组开停机状态，展开为 15min 明细。"""
        raw = self._get_data(start_date, end_date, "realTimeMachineRunStopStatus")
        df_daily = pd.DataFrame(raw)
        if df_daily.empty:
            return pd.DataFrame()

        rows = []
        for _, row in df_daily.iterrows():
            date_str = row.get("time")
            date_base = pd.to_datetime(date_str).date()
            content = row.get("content") or []
            for site in content:
                if not isinstance(site, dict):
                    continue
                site_name = site.get("siteName")
                for mach in site.get("machineList", []) or []:
                    machine_name = mach.get("machineName")
                    for s in mach.get("statusList", []) or []:
                        t_str = s.get("time")
                        status = s.get("status")
                        if not t_str:
                            continue
                        try:
                            t = datetime.strptime(t_str, "%H:%M").time()
                        except Exception:
                            continue
                        ts = datetime.combine(date_base, t)
                        rows.append(
                            {
                                "timestamp": ts,
                                "siteName": site_name,
                                "machineName": machine_name,
                                "status": status,
                            }
                        )
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        result.set_index("timestamp", inplace=True)
        return result.sort_index()

    # ---- 五类：电网设备停运情况及其影响（list[dict] 事件） ----

    def deviceStopSituation(self, start_date, end_date):
        """电网设备停运情况及其影响，展开为事件明细。"""
        raw = self._get_data(start_date, end_date, "deviceStopSituation")
        df_daily = pd.DataFrame(raw)
        if df_daily.empty:
            return pd.DataFrame()

        rows = []
        for _, row in df_daily.iterrows():
            day = row.get("time")
            content = row.get("content") or []
            for item in content:
                if not isinstance(item, dict):
                    continue
                rec = {"day": pd.to_datetime(day)}
                rec.update(item)
                rows.append(rec)
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        # 如果有 powerOutageTime 字段，优先用它做索引
        if "powerOutageTime" in result.columns:
            try:
                result["powerOutageTime"] = pd.to_datetime(result["powerOutageTime"])
                result.set_index("powerOutageTime", inplace=True)
            except Exception:
                result.set_index("day", inplace=True)
        else:
            result.set_index("day", inplace=True)
        return result.sort_index()

# --------------------------- 调试：打印 dataType 原始结构（已注释保留） ---------------------------
# 需要时可去掉每行开头的 “# ” 后，在交互环境中调用：
#   loader.debug_datatype("waterPowerWeekForecast", auto_search=True, max_rows=1)
# ------------------------------------------------------------------------------------------------
# def debug_datatype(self,
#                    data_type,
#                    start_date=None,
#                    end_date=None,
#                    max_rows: int = 1,
#                    auto_search: bool = True,
#                    search_window_days: int = 7,
#                    max_search_days: int = 365,
#                    truncate_len: int = 200):
#     # 打印某个 dataType 的原始返回结构（仅打印，不返回）
#     print("\n" + "=" * 80)
#     window_days = max(1, search_window_days)
# 
#     if auto_search or not (start_date and end_date):
#         end_dt = datetime.now().date() - timedelta(days=1)
#         found = False
#         for offset in range(0, max_search_days, window_days):
#             end_candidate = end_dt - timedelta(days=offset)
#             start_candidate = end_candidate - timedelta(days=window_days - 1)
#             start_str = start_candidate.strftime("%Y-%m-%d")
#             end_str = end_candidate.strftime("%Y-%m-%d")
# 
#             raw = self._get_data(start_str, end_str, data_type)
#             raw_df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
#             if not raw_df.empty:
#                 start_date, end_date = start_str, end_str
#                 found = True
#                 break
#         if not found:
#             print(f"调试 dataType = {data_type} | 未在最近 {max_search_days} 天内找到数据")
#             return
#     else:
#         raw = self._get_data(start_date, end_date, data_type)
#         raw_df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
# 
#     print(f"调试 dataType = {data_type} | 时间区间: {start_date} → {end_date}")
#     print("=" * 80)
#     print(f"\n原始返回类型: {type(raw_df)}")
#     print(f"DataFrame 形状: {raw_df.shape}")
#     print(f"列名: {list(raw_df.columns)}")
# 
#     if raw_df.empty:
#         print("⚠️ DataFrame 为空（该时间段内无数据或列缺失）")
#         return
# 
#     print(f"\n前 {max_rows} 行原始数据（简要）:")
#     print(raw_df.head(max_rows))
# 
#     if "content" in raw_df.columns:
#         print(f"\ncontent 列前 {max_rows} 行的结构:")
#         for idx, val in raw_df["content"].head(max_rows).items():
#             print(f"\n行 {idx}: 类型={type(val)}")
#             def fmt(v):
#                 s = repr(v)
#                 return s if len(s) <= truncate_len else s[:truncate_len] + "..."
#             if isinstance(val, (list, tuple)):
#                print(f"  列表/元组，长度={len(val)}")
#                if val:
#                    print(f"  第一个元素类型={type(val[0])}")
#                    print(f"  第一个元素值={fmt(val[0])}")
#             elif isinstance(val, dict):
#                print(f"  字典，键集合={list(val.keys())[:10]}")
#                print(f"  字典内容={fmt(val)}")
#             else:
#                print(f"  值={fmt(val)}")
#     else:
#         print("\n⚠️ 不存在 content 列")


# if __name__ == "__main__":
#     # 简单检验：随机挑选几个封装后的方法，打印 15min 粒度的结果样例
#     loader = ShandongMoreData()

#     # 可按需调整时间区间（尽量选择近期有数据的时间段）
#     start_date = "2020-01-01"
#     end_date = "2025-11-20"

#     tests = [
#         ("overhaulCapacity", loader.overhaulCapacity),
#         ("waterPowerWeekForecast", loader.waterPowerWeekForecast),
#         ("newEnergyWeekForecast", loader.newEnergyWeekForecast),
#         ("loadRegulationWeekForecast", loader.loadRegulationWeekForecast),
#         ("totalPowerWeekForecast", loader.totalPowerWeekForecast),
#         ("loadRegulationReal", loader.loadRegulationReal),
#         ("totalPowerReal", loader.totalPowerReal),
#         ("totalPowerPreForecast", loader.totalPowerPreForecast),
#         ("loadRegulationPreForecast", loader.loadRegulationPreForecast),
#         ("dayAheadSourceClearElecUnits", loader.dayAheadSourceClearElecUnits),
#         ("realTimeSourceClearElecUnits", loader.realTimeSourceClearElecUnits),
#         ("dailyUnitMaximumPower", loader.dailyUnitMaximumPower),
#         ("realTimeMachineRunStopStatus", loader.realTimeMachineRunStopStatus),
#         ("deviceStopSituation", loader.deviceStopSituation),
#     ]

#     for name, func in tests:
#         print("\n" + "=" * 60)
#         print(f"{name} | {start_date} → {end_date}")
#         print("=" * 60)
#         try:
#             df = func(start_date, end_date)
#             if isinstance(df, pd.DataFrame) and not df.empty:
#                 # 估计时间粒度（前 8 个点）
#                 freq = None
#                 if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 8:
#                     try:
#                         freq = pd.infer_freq(df.index[:8])
#                     except Exception:
#                         freq = None
#                 print(f"shape={df.shape}, inferred_freq={freq}")
#                 print(df.head(5))
#             else:
#                 print("empty dataframe or invalid result")
#         except Exception as e:
#             print(f"error: {e}")
    
    
    

  
   

    











    

    





    

    


    


    




