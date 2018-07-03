import os, sys, pprint
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import time, datetime
from os.path import join
import matplotlib.pyplot as plt

home_dir = os.path.expanduser('~')
printer = pprint.PrettyPrinter(indent=3)

band_list = ['B_band', 'G_band', 'R_band', 'NIR_band', 'SWIR1', 'SWIR2']

def truncate_data(data_series, start_day, end_day):
    '''truncate time and extract valid points only'''
    return [(dp[0], dp[1]) for dp in data_series if
            dp[0]>=start_day and dp[0]<=end_day
            and dp[1]>0 and dp[1]<1]

def fill_time_with_nan(data_series, start_day, end_day):
    '''convert invalid data and missing data to nan'''
    '''should be used AFTER remove duplicated'''
    day_list = []
    day_list.append(start_day)
    next_day = datetime.datetime.strptime(start_day, '%Y%m%d')
    '''strptime 把‘20120301’时间转换为datetime.datetime(2012, 3, 1, 0, 0)格式'''

    while next_day < datetime.datetime.strptime(end_day, '%Y%m%d'):
        next_day = next_day + datetime.timedelta(days=1)
        day_list.append(next_day.strftime('%Y%m%d'))
        '''strftime是strptime的逆运算'''

    data_days = [x[0] for x in data_series]
    valid_data = [(x[0], x[1]) for x in data_series if x[0] in data_days]
    '''be careful, only to process reflectance'''
    missing_days = [x for x in day_list if x not in data_days]

    new_series = []
    missing_data = [(d, np.nan) for d in missing_days]

    new_series += missing_data
    new_series += valid_data
    new_series.sort() # 按时间顺序排序
    return new_series

def remove_duplicates(data_series):
    '''should not call this function in mass production version'''
    new_series = []
    # ls_temp = []
    tmp_recorder = {}
    for dp in data_series:
        if dp[0] not in tmp_recorder.keys():
            tmp_recorder[dp[0]] = [dp[1]]
        else:
            tmp_recorder[dp[0]].append(dp[1])
            '''
            等效于：
            ls_temp.append(tmp_recorder[dp[0]])
            ls_temp.append(dp[1])
            tmp_recorder[dp[0]] = ls_temp
            '''
    for data_day in tmp_recorder.keys():
        # MARK: taking average of duplicated values
        new_series.append((data_day, np.mean(tmp_recorder[data_day])))
    new_series.sort()
    return new_series

def interpolate(data_series):
    datestr = list(zip(*(data_series)))[0]
    valuestr = list(zip(*(data_series)))[1]
    valuestr = pd.Series(valuestr)
    value_interp = valuestr.interpolate(method='linear', limit_direction='both')    # 将所有无效NAN值做插值
    # MARK: pay attention to limit_direction
    result = list(zip(datestr, value_interp))
    return result

def data_preprocessing(data_series, start_day, end_day):
    '''remove duplicates and fill data with NaN'''
    data_series = truncate_data(data_series, start_day, end_day)
    data_series = remove_duplicates(data_series)
    data_series = fill_time_with_nan(data_series, start_day, end_day)
    return data_series

def sg_the_fuck(data_series, win_len, poly_ord):
    datestr = list(zip(*(data_series)))[0]
    valuestr = list(zip(*(data_series)))[1]
    valuestr = pd.Series(valuestr)
    sg_result = savgol_filter(valuestr, window_length=win_len, polyorder=poly_ord)
    sged = list(zip(datestr, sg_result))
    return sged

def sg_ultimate(data_list, start_day, end_day):
    new_list = []
    for data_entry in data_list:
        NO_DATA = False
        new_entry = {}
        for band_type in band_list:
            data_series = data_entry[band_type]  # 选取特定band的数据
            data_series = truncate_data(data_series, start_day, end_day)  # 时间截断,过滤无效值
            data_series = remove_duplicates(data_series)  # 合并重复时间的数据
            data_series = fill_time_with_nan(data_series, start_day, end_day)  # 填补未观测的时间列表
            data_series = interpolate(data_series)  # 给整个时间列表插值
            data_series = sg_the_fuck(data_series, 33, 2)  # 数据通过SG滤波

            # check empty data
            all_data = [x[1] for x in data_series]
            if np.isnan(sum(all_data)):
                print('No data point')
                NO_DATA = True
                break  # 循环中断后，循环参数的值保持不变
            new_entry[band_type] = data_series
        if NO_DATA:
            continue
        new_list.append(new_entry)
    return new_list

'''

if __name__ == '__main__':
    sample_path = join(home_dir, 'data_pool', 'waterfall_data', '2016_Corn_2000_20180614072100_extractor_results.npz')
    data_list = np.load(sample_path)['arr_0']
    # data_list = sg_ultimate(data_list, '20160401', '20160627')
    # printer.pprint(data_list[:3])
    # print(len(data_list))
    start_day = '20160401'
    end_day = '20161001'


    test_series = data_list[0]['NIR_band']
    test_series.append(('20160520', -1))
    test_series.append(('20160504', 0.5))
    test_series.sort() # 按照时间顺序排序
    # printer.pprint(test_series)
    # result = remove_duplicates(test_series)
    new_series = truncate_data(test_series, start_day, end_day) # 选取制定时间段且有效的数据
    new_series = remove_duplicates(new_series)
    new_series = fill_time_with_nan(new_series, start_day, end_day)
    # printer.pprint(new_series)
    inter_series = interpolate(new_series)
    x1 = range(len(inter_series))
    y1 = [x[1] for x in inter_series]
    # printer.pprint(new_series)
    sg_series1 = sg_the_fuck(inter_series, 33, 2)
    x2 = range(len(sg_series1))
    y2 = [x[1] for x in sg_series1]
    sg_series2 = sg_the_fuck(inter_series, 33, 4)
    x3 = range(len(sg_series2))
    y3 = [x[1] for x in sg_series2]
    sg_series3 = sg_the_fuck(inter_series, 33, 3)
    x4 = range(len(sg_series3))
    y4 = [x[1] for x in sg_series3]
    # printer.pprint(new_series)
    # plt.figure(1)
    fig, ax = plt.subplots()
    ax.plot(x1, y1, 'k', label='Origin Data')
    ax.plot(x2, y2, 'r.-', label='SG(33,2)')
    ax.plot(x3, y3, 'g*-', label='SG(33,4)')
    ax.plot(x4, y4, 'b+-', label='SG(33,3)')
    lagend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.show()
'''

if __name__ == '__main__':
    beginClock = time.clock()
    home_dir = os.path.expanduser('~')
    sample_path = join(home_dir, 'data_pool', 'waterfall_data', '2016_Other_2000_20180614181524_extractor_results.npz')
    # npz文件包含了landsat7/8卫星对于地面同一位置的卫星图像，可能包含同一天来自两卫星的重复数值
    data_list = np.load(sample_path)['arr_0']
    start_day = '20160401'
    end_day = '20161001'
    new_list = sg_ultimate(data_list, start_day, end_day)
    # printer.pprint(new_list)
    endClock = time.clock()
    print((endClock - beginClock), 'sec')