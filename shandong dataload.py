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

    # overhaulCapacity 方法
    def overhaulCapacity(self, start_date, end_date):
        print("\n================= 调用 etide 接口测试 =================")
        print(f"请求时间区间: {start_date} → {end_date}")

        raw = self._get_data(start_date, end_date, 'overhaulCapacity')

        print("\n=== 原始返回数据类型 ===")
        print(type(raw))
        if isinstance(raw, pd.DataFrame):
            raw_df = raw.copy()
        else:
            raw_df = pd.DataFrame(raw)

        print("\n=== 原始返回数据长度 ===")
        print(len(raw_df))

        if raw_df.empty:
            print("\n❌ 未获取到任何数据！请检查日期或权限。\n")
            return pd.DataFrame(columns=['检修容量'])
        import json
        print("\n=== 原始返回内容（前 3 条）===\n")
        print(json.dumps(raw_df.head(3).to_dict(orient='records'), ensure_ascii=False, indent=2))

        if 'time' not in raw_df.columns or 'content' not in raw_df.columns:
            print("⚠️ 数据结构不符合预期，返回空 DataFrame")
            return pd.DataFrame(columns=['检修容量'])

        def extract_content(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (int, float, str)) or item is None:
                        return item
                    if isinstance(item, dict):
                        for _, v in item.items():
                            return v
                    if isinstance(item, (list, tuple)) and item:
                        return item[0]
                return value[0] if len(value) > 0 else None
            if isinstance(value, dict):
                for _, v in value.items():
                    return v
            return value

        df = raw_df.set_index('time')[['content']]
        df['content'] = df['content'].apply(extract_content)
        df.index = pd.to_datetime(df.index)
        df.columns = ['检修容量']

        print("\n=== Daily 检修容量数据（前 10 行）===")
        print(df.head(10))

        df_15min = self.convert_daily_to_15min(df)
        print("\n=== 15min 填充数据（前 10 行）===")
        print(df_15min.head(10))

        print("\n================= etide API 测试结束 =================\n")
        return df_15min

    # 通用调试：打印任意 dataType 的返回结构（只打印，不做转换）
    def debug_datatype(
        self,
        data_type,
        start_date=None,
        end_date=None,
        max_rows: int = 3,
        auto_search: bool = True,
        search_window_days: int = 7,
        max_search_days: int = 365,
    ):
        """
        调试辅助方法：查看某个 dataType 的原始返回结构。

        只做 print，不改变数据，便于你根据结构设计各自的处理逻辑。
        """
        print("\n" + "=" * 80)
        window_days = max(1, search_window_days)

        if auto_search or not (start_date and end_date):
            end_dt = datetime.now().date() - timedelta(days=1)
            found = False

            for offset in range(0, max_search_days, window_days):
                end_candidate = end_dt - timedelta(days=offset)
                start_candidate = end_candidate - timedelta(days=window_days - 1)

                start_str = start_candidate.strftime("%Y-%m-%d")
                end_str = end_candidate.strftime("%Y-%m-%d")

                raw = self._get_data(start_str, end_str, data_type)
                df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

                if not df.empty:
                    start_date = start_str
                    end_date = end_str
                    found = True
                    raw_df = df
                    break
            if not found:
                print(f"调试 dataType = {data_type} | 未在最近 {max_search_days} 天内找到数据")
                return
        else:
            raw = self._get_data(start_date, end_date, data_type)
            raw_df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

        print(f"调试 dataType = {data_type} | 时间区间: {start_date} → {end_date}")
        print("=" * 80)

        print(f"\n原始返回类型: {type(raw_df)}")

        print(f"DataFrame 形状: {raw_df.shape}")
        print(f"列名: {raw_df.columns.tolist()}")

        if raw_df.empty:
            print("⚠️ DataFrame 为空（这一段时间可能没有该 dataType 的数据）")
            return

        print(f"\n前 {max_rows} 行原始数据:")
        print(raw_df.head(max_rows))

        # 专门看一下 content 列的结构
        if "content" in raw_df.columns:
            print(f"\ncontent 列前 {max_rows} 行的详细结构:")
            for idx, val in raw_df["content"].head(max_rows).items():
                print(f"\n行 {idx}:")
                print(f"  类型: {type(val)}")
                print(f"  值: {repr(val)}")
                if isinstance(val, (list, tuple)) and val:
                    print(f"  第一个元素: {repr(val[0])} (类型: {type(val[0])})")
        else:
            print("\n⚠️ 不存在 content 列")


if __name__ == "__main__":
    loader = ShandongMoreData()

    # 可根据需要调整时间区间
    start_date = "2024-01-01"
    end_date = "2024-01-07"

    data_types = [
        ("水电含抽需", "waterPowerWeekForecast"),
        ("新能源", "newEnergyWeekForecast"),
        ("直调负荷", "loadRegulationWeekForecast"),
        ("全网负荷", "totalPowerWeekForecast"),
        ("实际负荷-直调负荷", "loadRegulationReal"),
        ("实际负荷-全网负荷", "totalPowerReal"),
        ("日前预计划-全网负荷", "totalPowerPreForecast"),
        ("日前预计划-直调负荷", "loadRegulationPreForecast"),
        ("日前各类电源出清电量及台数", "dayAheadSourceClearElecUnits"),
        ("实时各类电源出清电量及台数", "realTimeSourceClearElecUnits"),
        ("日机组最大出力", "dailyUnitMaximumPower"),
        ("实时各机组开停机状态", "realTimeMachineRunStopStatus"),
        ("电网设备停运情况及其影响", "deviceStopSituation"),
        ("煤电机组开停机台次和容量", "thermalMachineUnitsAndCapacity"),
        ("重要通道实际输电情况", "importantChannels"),
        ("实际运行输电断面约束情况", "transmissionSections"),
        ("重要线路与变压器平均潮流", "importantLinesAndTransformers"),
        ("发电机组检修计划执行情况", "generatorUnitMaintenancePlan"),
        ("输电设备检修计划执行", "transmissionEquipmentMaintenancePlan"),
        ("变电设备检修计划执行", "substationEquipmentMaintenancePlan"),
        ("日前发电侧出清均价", "powerGenerationSideInRecentDays"),
        ("日前市场各时段出清的断面约束及阻塞情况", "marketClearingAtVariousTimeIntervals"),
        ("市场干预情况原始日志", "originalLogOfMarketIntervention"),
        ("电网运行预测信息-阻塞信息", "powerGridOperationForecastInformation"),
        ("检修容量预测", "overhaulCapacity"),
        ("检修容量实际", "overhaulCapacityReal"),
    ]

    for zh_name, dtype in data_types:
        print("\n" + "#" * 100)
        print(f"{zh_name} ({dtype})")
        print("#" * 100)
        loader.debug_datatype(dtype, start_date, end_date)

    # 如果仍然想看 overhaulCapacity 转换后的结果，可取消下方注释
    # df = loader.overhaulCapacity(start_date, end_date)
    # print("\n========== 方法返回值 df.head() ==========")
    # print(df.head())
    
    
    

  
   

    











    

    





    

    


    


    

