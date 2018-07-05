import os,sys,pprint
import numpy as np
import pandas as pd
from  scipy.signal import savgol_filter
import time
import itertools as it
from datetime import datetime
# from numpy.core.multiarray import interp as compiled_interp

'''
interpolate the lost times that has no suitable images by given the start time and
end time  and filter the time  by the range of ('20170401','20170930')
'''
class SGFilter:

    def __init__(self, start_time, end_time, step = 1):
        self.start_time = start_time
        self.end_time = end_time
        self.ts = self.creat_time_series(start_time, end_time, step)
        self.ts_tm_yday = list(map(self.ymd_to_jd, self.ts))

    def ymd_to_jd(self, time_str, fmt = '%Y%m%d'):
        dt = datetime.strptime(time_str, fmt)
        tt = dt.timetuple()
        return tt.tm_yday


    def creat_time_series(self, startime, endtime, step=1):
        '''
        Function:
            to get the date string from startime to endtime
        Input:
            startime and endtime like this '20170612' , '20170615'
        Output:
            return:like input above, output is ['20170612', '20170613', '20170614', '20170615']
        '''
        # get the step
        if step == 1:
            step = '1D'
        else:
            step = str(step)+'D'

        time_series = []
        tar_timeseries = []
        for time in pd.date_range(startime,endtime,freq = step):
            time_series.append(pd.date_range(time,periods = 1, freq = '1D').strftime("%Y%m%d").tolist()[0])
        tar_timeseries = time_series #??why
        tar_timeseries.sort()

        return tar_timeseries

    def interp_by_ts(self, pixel_by_bands):
        '''
        Returns
        -------
        one-dim-arr
            result will flatten to one dim
        '''
        # # Remove duplicate date: 1000 * 1000 about 2 sec
        # for band_name in pixel_by_bands:
        #     mean_ret = []
        #     band_val = pixel_by_bands[band_name]

        #     for date, values in it.groupby(band_val,lambda x: x[0]):
        #         total = 0
        #         num = 0
        #         for (_, v) in values:
        #             if v <= 10000 and v > 0:
        #                 total += v
        #                 num += 1
        #         if num != 0: mean_ret.append((date, total/(num*10000)))
        #     if mean_ret == [] : return None
        #     else: pixel_by_bands[band_name] = mean_ret

        # make sure first item has value for interpolate
        band_names =['R_band', 'G_band', 'B_band', 'NIR_band', 'SWIR1', 'SWIR2'] # need have this order
        all_with_interp = []
        for band_name in band_names:
            band_vals = pixel_by_bands[band_name]

            if band_vals != [] :
                band_vals = np.asarray(band_vals).T # band_vals => [[date, val],[date,val]]
                date_list = band_vals[0]
                date_vals = band_vals[1]
                # with_interp = compiled_interp(self.ts_tm_yday, date_list, date_vals)
                with_interp = np.interp(self.ts_tm_yday, date_list, date_vals)
                all_with_interp.append(with_interp) # 插值
            else:
                return None # invalid pixel, return none

        return all_with_interp

    def run_sg(self, pixel_bands_by_date, window_len = 33, polyorder = 2, axis = 1):
        '''
        Args:
        -----
            pixel_bands_by_date: [pixel_data,pixel_data,...]
                pixel_data: [r1,r2,r3..,g1,g2,g3...,b1,b2,b3...] , all_sorted_band_val_by_date
            window_len, polyorder : 33 ,2 needed since ml model use this training
            axis:  should be set properly with the input dim in the date_series

        Return:
        ----
            with same dim
        '''
        sg_res = savgol_filter(pixel_bands_by_date, window_length = window_len, polyorder = polyorder, axis= axis)

        return sg_res
