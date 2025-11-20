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

    # 直接请求原始数据
    raw = self._get_data(start_date, end_date, 'overhaulCapacity')

    print("\n=== 原始返回数据类型 ===")
    print(type(raw))

    print("\n=== 原始返回数据长度 ===")
    print(len(raw))

    # 如果没有数据
    if not raw:
        print("\n❌ 未获取到任何数据！请检查日期或权限。\n")
        return

    # 打印前 3 条完整 JSON
    import json
    print("\n=== 原始返回内容（前 3 条）===\n")
    print(json.dumps(raw[:3], ensure_ascii=False, indent=2))

    # 打印每条的 time 和 content
    print("\n=== 每条记录的 time + content（前 3 条）===")
    for i, item in enumerate(raw[:3]):
        print(f"\n--- 第 {i+1} 条 ---")
        print(f"time: {item.get('time')}")
        print(f"content 类型: {type(item.get('content'))}")
        print(f"content 值: {repr(item.get('content'))}")

    print("\n================= etide API 测试结束 =================\n")

    # 返回原始数据即可
    return raw


if __name__ == "__main__":
    loader = ShandongMoreData()
    loader.overhaulCapacity("2024-01-01", "2024-01-03")
   
    df = loader.overhaulCapacity("2024-01-01", "2024-01-03")

    print("\n========== 方法返回值 df.head() ==========")
    print(df.head())
    
    

  
   

    











    

    





    

    


    


    
