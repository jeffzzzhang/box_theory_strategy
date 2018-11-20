# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:37:37 2017
由双均线策略修改得到的箱型策略, box theory
三个买入点：20日>60日；收盘价>60日；刚刚盈利7%平仓之后
两个卖出点：亏损1%；盈利3%
Box theory stems from dual MA(moving average) strategy.
entry points: price of 20-day MA > 60-day MAC; current closing price > price of 60-day MA; closing a position after 7% profit
exit points: loss 1%; gain 3%
@author: Jeff
"""
import numpy as np
import pandas as pd
import time,os,pickle,math

start_time = time.time()
def MovingAverage(number,closing_price):
    MA_n = np.array(closing_price)
    for stock_id in range(len(closing_price.columns)):
        this_stock_data = closing_price[closing_price.columns[stock_id]]
        for i in range(len(this_stock_data.index)):
            if type(this_stock_data[i]) == str:
                if this_stock_data[i] != '--':
                    continue
                else:
                    MA_n[i,stock_id] = '--'
                    continue
            else:
                if i < 2 + (number-1): # assume the head 2 lines are stock code and name
                    MA_n[i,stock_id] = '--'
                else:
                    tmp = list(this_stock_data[i-number+1:i+1])
                    tmp_cntr = 0
                    for j in range(number):
                        if type(tmp[j]) == str:
                            tmp_cntr = 1
                            break
                    if tmp_cntr == 1:
                        MA_n[i,stock_id] = '--'
                        continue
                    MA_n[i,stock_id] = sum(tmp)/number # this_stock_data[i-number:i].mean()
    MA = pd.DataFrame(MA_n,index=closing_price.index,columns=closing_price.columns)
    return MA

def ma_cross(ma_small,ma_big,stock_list):
    '''
    是否长线MA大于短线MA的时间序列
    '''    
    compare_array = np.array(ma_big[stock_list])
    starting_index = 0
    for i in range(len(ma_big.index)):
        if type(ma_big.iloc[i,0]) == str:
            continue
        else:
            starting_index = i
            break
    for stock in range(len(stock_list)):
        tmp_compare_bit = []
        tmp_compare_bit = list(np.sign(ma_big.ix[starting_index:,stock_list[stock]]-ma_small.ix[starting_index:,stock_list[stock]]))
        compare_array[starting_index:,stock] = tmp_compare_bit
        
    return pd.DataFrame(compare_array,index=ma_big.index,columns=stock_list)
    
def buy_sell(ma_cross_results,ma60,closing_price,opening_price):
    buy_sell_action = np.array(ma_cross_results)
    buy_sell_action[2:,:] = '--'
     #假设开盘买入，如果买入后任一天的收盘价比开盘买入价高7%或者低3%，则下一交易日开盘卖出
    
    for j in range(len(ma_cross_results.columns)):
        cost = None
        redeem = None
        for i in range(2,len(ma_cross_results.index)):
            if cost != None:
                if closing_price.iloc[i,j] >= cost * 1.03:
                    if buy_sell_action[i-1,j] == '--' or buy_sell_action[i-1,j] == 'sell':
                        buy_sell_action[i,j] = '--'
                        redeem = 'no'
                        cost = None
                    elif buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                        buy_sell_action[i,j] = 'sell'
                        redeem = 'yes' # 某日收盘价盈利七趴，则第二个交易日开始即卖出，第三个交易日再买入
                        cost = None
                elif closing_price.iloc[i,j] <= cost * 0.99:
                    if buy_sell_action[i-1,j] == '--' or buy_sell_action[i-1,j] == 'sell':
                        buy_sell_action[i,j] = '--'
                        redeem = 'no'
                        cost = None
                    elif buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                        buy_sell_action[i,j] = 'sell'
                        redeem = 'no'
                        cost = None
                else:
                    if buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                        buy_sell_action[i,j] = 'hold'
                        redeem = 'no'
            elif cost == None: #判断各种买入的情况
                if redeem == 'yes':
                    if buy_sell_action[i-1,j] != 'sell':
                        raise ValueError
                    if buy_sell_action[i-1,j] == 'sell':
                        buy_sell_action[i,j] = 'buy'
                        if i < len(opening_price.index)-1:
                            cost = opening_price.ix[i+1,ma_cross_results.columns[j]]
                            redeem = 'no'
                        else:
                            cost = None
                            redeem = 'no'
                else: # redeem = NO 下面开始判断其他买入条件
                    if ma_cross_results.iloc[i,j] == -1:
                        #######################################################
                        # 1st: 如果和前一天的ma60-ma20值相同，则没有什么，执行else：里面的动作
                        if ma_cross_results.iloc[i-1,j] == -1:
                            if buy_sell_action[i-1,j] == 'hold' or buy_sell_action[i-1,j] == 'buy':
                                buy_sell_action[i,j] = 'hold'
                                if cost == None:
                                    print(5)
                                    raise ValueError
                            elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                                buy_sell_action[i,j] = '--'
                                if cost != None:
                                    print(6)
                                    raise ValueError
                        # 2nd: 如果前一天ma60-ma20是1，则买入
                        elif ma_cross_results.iloc[i-1,j] == 1:
                            if buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                                buy_sell_action[i,j] = 'hold'
                                if cost == None:
                                    print(1)
                                    raise ValueError
                            elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                                buy_sell_action[i,j] = 'buy'
                                if i < len(opening_price.index)-1:
                                    cost = opening_price.ix[i+1,ma_cross_results.columns[j]]
                                else:
                                    cost = None
                        elif ma_cross_results.iloc[i-1,j] == 0:
                        # 3rd: 如果T-1是0，则再往前退，直到找到不是0的那一天，
                            tmp_cntr = 1
                            tmp_bit = None
                            while (ma_cross_results.iloc[i-tmp_cntr,j] == 0):
                                tmp_cntr += 1
                                tmp_bit = ma_cross_results.iloc[i-tmp_cntr,j]

                            if type(tmp_bit) == str:
                                print(7)
                                raise ValueError
                            if tmp_bit == -1:
                                if buy_sell_action[i-1,j] == 'hold' or buy_sell_action[i-1,j] == 'buy':
                                    buy_sell_action[i,j] = 'hold'
                                    if cost == None:
                                        print(9)
                                        raise ValueError
                                elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                                    buy_sell_action[i,j] = '--'
                                    if cost != None:
                                        print(10)
                                        raise ValueError
                            else:
                               # tmp_bit = 1 则穿越，要买入
                               if buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                                   buy_sell_action[i,j] = 'hold'
                                   if cost == None:
                                       print(8)
                                       raise ValueError
                               elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                                   buy_sell_action[i,j] = 'buy'
                                   if i < len(opening_price.index)-1:
                                       cost = opening_price.ix[i+1,ma_cross_results.columns[j]]
                                   else:
                                       cost = None
                            
                        #######################################################
                    elif i > 60 and closing_price.ix[i,ma_cross_results.columns[j]] > ma60.ix[i,ma_cross_results.columns[j]]:
                        if buy_sell_action[i-1,j] == 'buy' or buy_sell_action[i-1,j] == 'hold':
                            buy_sell_action[i,j] = 'hold'
                            if cost == None:
                                print(2)
                                raise ValueError
                        elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                            buy_sell_action[i,j] = 'buy'
                            if i < len(opening_price.index)-1:
                                cost = opening_price.ix[i+1,ma_cross_results.columns[j]]
                            else:
                                cost = None
                    else:
                        if buy_sell_action[i-1,j] == 'hold' or buy_sell_action[i-1,j] == 'buy':
                            buy_sell_action[i,j] = 'hold'
                            if cost == None:
                                print(3)
                                raise ValueError
                        elif buy_sell_action[i-1,j] == 'sell' or buy_sell_action[i-1,j] == '--':
                            buy_sell_action[i,j] = '--'
                            if cost != None:
                                print(4)
                                raise ValueError
                            
    buy_sell_action = pd.DataFrame(buy_sell_action,index=ma_cross_results.index,columns=ma_cross_results.columns)        
    return buy_sell_action
    
def gain_and_loss(buy_sell_action,opening_price):
    # 根据buy_sell_based_on_ma_cross的结果和开盘价来计算损失和收益,2017.06.16 加入手续费，买卖各千分之五
    position_status = np.array(buy_sell_action)
    stock_list = buy_sell_action.columns
    money_allocation = 100000
    remaining_cap = np.array(buy_sell_action)
    starting_index = None
    for i in range(len(buy_sell_action.index)):
        if type(buy_sell_action.index[i]) != str:
            starting_index = i
            break
    remaining_cap[starting_index:,:] = np.zeros((len(buy_sell_action.index)-starting_index,len(buy_sell_action.columns))) + money_allocation
    remaining_cap = pd.DataFrame(remaining_cap,index=buy_sell_action.index,columns=buy_sell_action.columns)
    
    for stock in range(len(stock_list)):
        hands = None
        buying_cost = None
        for i in range(2,len(buy_sell_action.index)):
            if buy_sell_action.iloc[i,stock] == 'buy':
                if buy_sell_action.iloc[i-1,stock] == '--':
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] == 'sell':
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] + 0.995*hands * opening_price.ix[i,stock_list[stock]] * 100
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
            elif buy_sell_action.iloc[i,stock] == 'hold':
                if buy_sell_action.iloc[i-1,stock] == 'buy':
                    hands = math.trunc(remaining_cap.iloc[i,stock]/(opening_price.ix[i,stock_list[stock]]*100))
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] *100
                    buying_cost = hands * opening_price.ix[i,stock_list[stock]] *100
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - 1.005*buying_cost
                else: # hold yesterday, hold today
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] * 100
                    # remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - buying_cost
            elif buy_sell_action.iloc[i,stock] == 'sell':
                if buy_sell_action.iloc[i-1,stock] == 'buy':
                    hands = math.trunc(remaining_cap.iloc[i,stock]/(opening_price.ix[i,stock_list[stock]]*100))
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] *100
                    buying_cost = hands * opening_price.ix[i,stock_list[stock]] *100
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - 1.005*buying_cost
                elif buy_sell_action.iloc[i-1,stock] == 'hold':
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] * 100
                else:
                    print('check here')
            elif buy_sell_action.iloc[i,stock] == '--':
                if buy_sell_action.iloc[i-1,stock] == 'sell':
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] + 0.995*hands * opening_price.ix[i,stock_list[stock]] * 100
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] == '--':
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] != '--' and type(buy_sell_action.iloc[i-1,stock]) == str:
                    position_status[i,stock] = 0
                    hands = None 
                    buying_cost = None
    return pd.DataFrame(position_status,index=buy_sell_action.index,columns=buy_sell_action.columns),remaining_cap #两个array在同一天的值相加即为总资产

if __name__ == '__main__':
    os.chdir('C:\\Users\\LH\\Desktop\\工作日志\\2017年6月第2周')
    ori_closing_price = pd.read_excel('个股日收盘价20160901till20170614.xlsx')
    ori_opening_price = pd.read_excel('个股日开盘价20160901till20170614.xlsx')
    tobe_eli = []
    for j in ori_closing_price.columns:
        for i in range(2,len(ori_closing_price.index)):
            if ori_closing_price.ix[i,j] == '--':
                tobe_eli.append(j)
                break
    closing_price = ori_closing_price.drop(tobe_eli,axis=1)
    opening_price = ori_opening_price.drop(tobe_eli,axis=1)
    ma20 = MovingAverage(20,closing_price)
    ma60 = MovingAverage(60,closing_price)
    ma5 = MovingAverage(5,closing_price)
    # ma5 = MovingAverage(5,closing_price)
    # ma5 = pickle.load(open('C:/Users/LH/Desktop/工作日志/2017年6月第1周/MA5_2016till20170606.txt','rb'))
    # ma20 = pickle.load(open('C:/Users/LH/Desktop/工作日志/2017年6月第1周/MA20_2016till20170606.txt','rb'))
    # ma60 = pickle.load(open('C:/Users/LH/Desktop/工作日志/2017年6月第1周/MA60_2016till20170606.txt','rb'))
    name = '002508.SZ'
    tmp_list = [name]#['600900.SH','600009.SH','600104.SH']
    stock_list = opening_price.columns[:10]
    results_ma_cross = ma_cross(ma20,ma60,tmp_list)
    results_buy_sell = buy_sell(results_ma_cross,ma60,closing_price,opening_price)
    cap_on_stock,cap_remaining = gain_and_loss(results_buy_sell,opening_price)# cap_on_stock,cap_remaining的同一列相加就得到某一天的总资产，可算得收益率等
    tmp_sum = np.array(cap_on_stock.iloc[2:,:] + cap_remaining.iloc[2:,:])
    tmp_sum = sum(tmp_sum.T)
    import matplotlib.pyplot as plt
    tmp_name = 'C:/Users/LH/Desktop/工作日志/2017年6月第2周/'+name+'.png'
    plt.plot(opening_price.index[61:],tmp_sum[59:]),plt.ylabel('total asset'),plt.title(name),plt.grid(b=None),plt.savefig(tmp_name,dpi=300)
    print('time collapsed: ',time.time()-start_time,' seconds')
