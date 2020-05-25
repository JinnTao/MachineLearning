# -*- coding:utf-8 -*-

from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqSim
from tqsdk.tools import DataDownloader

api = TqApi(TqSim())
save_path = "History_data/"

download_tasks = {}
# # 下载从 2018-01-01 到 2018-09-01 的 SR901 日线数据
# download_tasks["SR_daily"] = DataDownloader(api, symbol_list="CZCE.SR901", dur_sec=24*60*60,
#                     start_dt=date(2018, 1, 1), end_dt=date(2018, 9, 1), csv_file_name= save_path + "SR901_daily.csv")
#
# # 下载从 2017-01-01 到 2018-09-01 的 rb主连 5分钟线数据
# download_tasks["rb_5min"] = DataDownloader(api, symbol_list="KQ.m@SHFE.rb", dur_sec=5*60,
#                     start_dt=date(2017, 1, 1), end_dt=date(2018, 9, 1), csv_file_name=save_path + "rb_5min.csv")
#
# # 下载从 2018-01-01凌晨6点 到 2018-06-01下午4点 的 cu1805,cu1807,IC1803 分钟线数据，所有数据按 cu1805 的时间对齐
# # 例如 cu1805 夜盘交易时段, IC1803 的各项数据为 N/A
# # 例如 cu1805 13:00-13:30 不交易, 因此 IC1803 在 13:00-13:30 之间的K线数据会被跳过
# download_tasks["cu_min"] = DataDownloader(api, symbol_list=["SHFE.cu1805", "SHFE.cu1807", "CFFEX.IC1803"], dur_sec=60,
#                     start_dt=datetime(2018, 1, 1, 6, 0 ,0), end_dt=datetime(2018, 6, 1, 16, 0, 0), csv_file_name="cu_min.csv")
#
# # 下载从 2018-05-01凌晨0点 到 2018-06-01凌晨0点 的 T1809 盘口Tick数据
# download_tasks["T_tick"] = DataDownloader(api, symbol_list=["CFFEX.T1809"], dur_sec=0,
#                     start_dt=datetime(2018, 5, 1), end_dt=datetime(2018, 6, 1), csv_file_name=save_path + "T1809_tick.csv")

# JD
download_tasks["JD_daily"] = DataDownloader(api, symbol_list="DCE.jd1909", dur_sec=24*60*60,
                    start_dt=date(2018, 6, 1), end_dt=date(2020, 12, 31), csv_file_name= save_path + "JD1909_daily.csv")

ls = [v["underlying_symbol"] for k,v in api._data["quotes"].items() if k.startswith("KQ.m@jd")]


# # 使用with closing机制确保下载完成后释放对应的资源
# with closing(api):
#     while not all([v.is_finished() for v in download_tasks.values()]):
#         api.wait_update()
#         print("progress: ", { k:("%.2f%%" % v.get_progress()) for k,v in download_tasks.items() })