from datetime import datetime, timedelta
import calendar

def int_to_date(date_int):
    return datetime.strptime(str(date_int), "%Y%m%d").date()

def date_to_int(date_obj):
    return int(date_obj.strftime("%Y%m%d"))

def add_months(source_date, months):
    """ 增加月份，不會爆掉月底 """
    month = source_date.month - 1 + months
    year = source_date.year + month // 12
    month = month % 12 + 1
    day = min(source_date.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day).date()

def get_episode_range_rolling(base_start_int, episode_index, window_months=12, step_months=3):
    """
    base_start_int: 初始日期 int，例如 20000101
    episode_index: 第幾個 episode（從 0 開始）
    window_months: 每段訓練視窗的長度（月）
    step_months: 每次往後移幾個月（預設 3）
    """
    start_date = add_months(int_to_date(base_start_int), episode_index * step_months)
    end_date = add_months(start_date, window_months) - timedelta(days=1)

    return date_to_int(start_date), date_to_int(end_date)
