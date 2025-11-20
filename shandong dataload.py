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
        raw_data = pd.DataFrame(self._get_data(start_date, end_date, 'overhaulCapacity'))
        
        # 调试：检查原始数据
        print("=" * 60)
        print("调试：原始API返回数据")
        print("=" * 60)
        print(f"数据形状: {raw_data.shape}")
        print(f"列名: {raw_data.columns.tolist()}")
        if len(raw_data) > 0:
            print("\n前3行原始数据:")
            print(raw_data.head(3))
            if 'content' in raw_data.columns:
                print("\ncontent 列的前3个值:")
                for idx, val in raw_data['content'].head(3).items():
                    print(f"  行 {idx}: 类型={type(val)}, 值={repr(val)}")
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        print(f"    第一个元素: {repr(val[0])}, 类型={type(val[0])}")
        print()
        
        if raw_data.empty or 'time' not in raw_data.columns or 'content' not in raw_data.columns:
            print("未获取到检修容量数据，返回空 DataFrame")
            return pd.DataFrame()

        raw_df = raw_data.set_index('time')[['content']]

        # 调试：检查提取前的值
        print("=" * 60)
        print("调试：提取 content 值")
        print("=" * 60)
        print("提取前的前3个 content 值:")
        for idx, val in raw_df['content'].head(3).items():
            print(f"  {idx}: {repr(val)}")
        print()

        # content 是 list，所以取第一个
        # 改进：处理多种可能的数据结构
        def extract_content(x):
            if isinstance(x, (list, tuple)) and len(x) > 0:
                first = x[0]
                # 如果第一个元素也是列表，再取一层
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return first[0]
                # 如果第一个元素是字典，取第一个值
                elif isinstance(first, dict) and first:
                    return list(first.values())[0]
                else:
                    return first
            elif isinstance(x, dict) and x:
                return list(x.values())[0]
            else:
                return x
        
        raw_df['content'] = raw_df['content'].apply(extract_content)
        
        # 调试：检查提取后的值
        print("提取后的前3个 content 值:")
        for idx, val in raw_df['content'].head(3).items():
            print(f"  {idx}: {repr(val)}, 类型={type(val)}")
        print()

        raw_df.index = pd.to_datetime(raw_df.index)
        raw_df.columns = ['检修容量']

        # ⭐ 打印 daily 结果
        print("=" * 60)
        print("Daily 检修容量数据")
        print("=" * 60)
        print(raw_df)
        print(f"非零值数量: {(raw_df['检修容量'] != 0).sum() if not raw_df.empty else 0}")

        # ⭐ 转换成 15 分钟数据
        df_15min = self.convert_daily_to_15min(raw_df)

        print("\n" + "=" * 60)
        print("15 分钟填充后的检修容量")
        print("=" * 60)
        print(df_15min.head(10))
        print(f"非零值数量: {(df_15min['检修容量'] != 0).sum() if not df_15min.empty else 0}")

        return df_15min



if __name__ == "__main__":
    loader = ShandongMoreData()

   
    df = loader.overhaulCapacity("2024-01-01", "2024-01-03")

    print("\n========== 方法返回值 df.head() ==========")
    print(df.head())
    
    

  
   

    











    

    





    

    


    

    