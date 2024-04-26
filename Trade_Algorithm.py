import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def straddle_CE_and_PE(data ,expiry_week ,time):
    data_for_time = data[data['Time'] == time]
    data_within_expiry = data_for_time[data_for_time['Contract_Weekly'] == expiry_week]
    call_options = data_within_expiry[data_within_expiry['Type'] == 'CE']
    put_options = data_within_expiry[data_within_expiry['Type'] == 'PE']

    return (call_options,put_options)

def get_CE_and_PE(data ,expiry_week):
    data_within_expiry = data[data['Contract_Weekly'] == expiry_week]
    call_options = data_within_expiry[data_within_expiry['Type'] == 'CE']
    put_options = data_within_expiry[data_within_expiry['Type'] == 'PE']

    return (call_options,put_options)

def fetch_futures_price_by_date(dataframe):
    futures_price = dataframe[['Date','Close']]
    futures_price.reset_index(drop=True,inplace=True)
    return futures_price


def find_wing_strike(atm_price, options_data, percentage, Bool):
    if Bool == True:
        wing_strike  = atm_price * (1 + percentage)
    else:
        wing_strike  = atm_price * (1- percentage)
    
    strike_diff = abs(options_data['Strike'] - wing_strike)
    min_diff_index = strike_diff.idxmin()

    min_diff_row = options_data.loc[min_diff_index].copy()
    min_diff_row['Strike_Diff'] = strike_diff[min_diff_index]  

    return min_diff_row

def find_closest_strike(atm_price, options_data):
    strike_diff = abs(options_data['Strike'] - atm_price)

    min_diff_index = strike_diff.idxmin()


    min_diff_row = options_data.loc[min_diff_index].copy()
    min_diff_row['Strike_Diff'] = strike_diff[min_diff_index]

    return min_diff_row

def find_atm_call_put(futures_price, CE_data, PE_data):
    selected_call_options = []
    selected_put_options = []
    
    for date in futures_price['Date']:
        atm_price = futures_price[futures_price['Date'] == date]['Close'].iloc[0]
        
        closest_call_option = find_closest_strike(atm_price, CE_data[CE_data['Date'] == date])
        selected_call_options.append(closest_call_option)

        closest_put_option = find_closest_strike(atm_price, PE_data[PE_data['Date'] == date])
        selected_put_options.append(closest_put_option)

    selected_atm_call_data = pd.DataFrame(selected_call_options)
    selected_atm_put_data = pd.DataFrame(selected_put_options)

    return selected_atm_call_data,selected_atm_put_data

def find_wing_call_put(futures_price, CE_data, PE_data,percentage):
    wing_call_options = []
    wing_put_options = []
    
    for date in futures_price['Date']:
        atm_price = futures_price[futures_price['Date'] == date]['Close'].iloc[0]
        
        closest_call_option = find_wing_strike(atm_price, CE_data[CE_data['Date'] == date],percentage,True)
        wing_call_options.append(closest_call_option)

        closest_put_option = find_wing_strike(atm_price, PE_data[PE_data['Date'] == date],percentage,False)
        wing_put_options.append(closest_put_option)

    selected_wing_call_data = pd.DataFrame(wing_call_options)
    selected_wing_put_data = pd.DataFrame(wing_put_options)

    return selected_wing_call_data,selected_wing_put_data

def calculate_stop_loss_target(atm_call_premium,atm_put_premium,wing_call_premium,wing_put_premium,stop_loss_pct,target_pct):
    total_premium = atm_call_premium+atm_put_premium-(wing_call_premium+wing_put_premium)
    stop_loss = total_premium*(1+stop_loss_pct)
    target = total_premium*(1-target_pct)
    return stop_loss,target

def monitor_stop_loss_target(stop_loss,target,curr_premium):
    stop_loss_reached = False
    target_reached = False
    

    if curr_premium >= stop_loss:
        stop_loss_reached = True
        

    elif curr_premium <= target:
        target_reached = True

    return stop_loss_reached,target_reached

def monitor_position_data(historical_data,ticker,entry_date,expiry_date):
    entry_time = dt.datetime.strptime('10:29:59', '%H:%M:%S').time()
    expiry_time = dt.datetime.strptime('15:19:59', '%H:%M:%S').time()

    historical_data = historical_data[historical_data['Ticker'] == ticker]
  
    trade_entry_date = pd.to_datetime(entry_date).date()
    trade_expiry_date = pd.to_datetime(expiry_date).date()

    if trade_entry_date != trade_expiry_date:
        trade_data = historical_data[
            ((pd.to_datetime(historical_data['Date']).dt.date == trade_entry_date) & (historical_data['Time'].apply(lambda x: dt.datetime.strptime(x, '%H:%M:%S').time()) >= entry_time)) |
            ((pd.to_datetime(historical_data['Date']).dt.date > trade_entry_date) & (pd.to_datetime(historical_data['Date']).dt.date < trade_expiry_date)) |
            ((pd.to_datetime(historical_data['Date']).dt.date == trade_expiry_date) & (historical_data['Time'].apply(lambda x: dt.datetime.strptime(x, '%H:%M:%S').time()) <= expiry_time))
        ]
    else:
        trade_data = historical_data[
            (pd.to_datetime(historical_data['Date']).dt.date == trade_entry_date) & 
            (historical_data['Time'].apply(lambda x: dt.datetime.strptime(x, '%H:%M:%S').time()) >= entry_time) &
            (historical_data['Time'].apply(lambda x: dt.datetime.strptime(x, '%H:%M:%S').time()) <= expiry_time)
        ]

    trade_data.reset_index(drop=True,inplace=True)
    
    return trade_data

def monitor_call_straddle_data(historical_data_ce,straddle_data):
    trade_data_dict = {}
    for index , row in straddle_data.iterrows():
        ticker = row['Ticker']
        entry_date = row['Date']
        expiry_date = row['Expiry']
        trade_data_ce = monitor_position_data(historical_data_ce,ticker,entry_date,expiry_date)
        trade_data_dict[entry_date] = trade_data_ce
    return trade_data_dict
    
def monitor_put_straddle_data(historical_data_pe,straddle_data):
    trade_data_dict = {}
    for index , row in straddle_data.iterrows():
        ticker = row['Ticker']
        entry_date = row['Date']
        expiry_date = row['Expiry']
        trade_data_pe = monitor_position_data(historical_data_pe,ticker,entry_date,expiry_date)
        trade_data_dict[entry_date] = trade_data_pe
    return trade_data_dict

def monitor_premium(trade_dict_atm_CE, trade_dict_atm_PE, trade_dict_wing_CE, trade_dict_wing_PE, stop_lost_dict,target_dict):
    exit_data = []
    for (entry_date, trade_data_atm_CE) in trade_dict_atm_CE.items():
        # Checking the entry date exists in the PE trade data as well
        if (entry_date in trade_dict_atm_PE) and (entry_date in trade_dict_wing_CE) and (entry_date in trade_dict_wing_PE):
            # trade data for ATM CE and PE legs
            trade_data_atm_CE = trade_dict_atm_CE[entry_date]
            trade_data_atm_PE = trade_dict_atm_PE[entry_date]
            # trade data for Wing CE and PE legs
            trade_data_wing_CE = trade_dict_wing_CE[entry_date]
            trade_data_wing_PE = trade_dict_wing_PE[entry_date]
            trade_dict = {}
            for idx, row in trade_data_atm_CE.iterrows():
                date = row['Date']
                time_stamp = row['Time']
                exit_reached = False
                # Check if the timestamp exists in all other datasets for the same date
                if (time_stamp in trade_data_atm_PE[trade_data_atm_PE['Date'] == date]['Time'].values) and \
                        (time_stamp in trade_data_wing_CE[trade_data_wing_CE['Date'] == date]['Time'].values) and \
                        (time_stamp in trade_data_wing_PE[trade_data_wing_PE['Date'] == date]['Time'].values):
                    # Calculate premiums
                    atm_ce_premium = row['Close']
                    atm_pe_premium = trade_data_atm_PE[(trade_data_atm_PE['Date'] == date) & (trade_data_atm_PE['Time'] == time_stamp)]['Close'].values[0]
                    wing_ce_premium = trade_data_wing_CE[(trade_data_wing_CE['Date'] == date) & (trade_data_wing_CE['Time'] == time_stamp)]['Close'].values[0]
                    wing_pe_premium = trade_data_wing_PE[(trade_data_wing_PE['Date'] == date) & (trade_data_wing_PE['Time'] == time_stamp)]['Close'].values[0]
                    total_premium = atm_ce_premium + atm_pe_premium - wing_ce_premium - wing_pe_premium
                    
                    # Check if stop loss or target is reached
                    stop_loss_reached, target_reached = monitor_stop_loss_target(stop_lost_dict[entry_date], target_dict[entry_date], total_premium)

                    if stop_loss_reached or target_reached:
                        exit_reached = True
                        exit_datetime = f"{date} {time_stamp}"
                        exit_data.append({
                            'Ce_Ex_Date': exit_datetime,
                            'Pe_Ex_Date': exit_datetime,
                            'Ce_Short_Ex_Price': atm_ce_premium,
                            'Pe_Short_Ex_Price': atm_pe_premium,
                            'Ce_Long_Ex_Price': wing_ce_premium,
                            'Pe_Long_Ex_Price': wing_pe_premium,
                            'Exit_Premium': total_premium
                        })
                        break
                    else:
                        exit_datetime = f"{date} {time_stamp}"
                        trade_dict = {
                            'Ce_Ex_Date': exit_datetime,
                            'Pe_Ex_Date': exit_datetime,
                            'Ce_Short_Ex_Price': atm_ce_premium,
                            'Pe_Short_Ex_Price': atm_pe_premium,
                            'Ce_Long_Ex_Price': wing_ce_premium,
                            'Pe_Long_Ex_Price': wing_pe_premium,
                            'Exit_Premium': total_premium
                        }

            if exit_reached == False:
                exit_data.append(trade_dict)

    return exit_data

def pfolio_daily_drawdown(pct_ret):
    pct_ret = [x/ 100 for x in pct_ret]
    Pfolio_Total_Daily_Drawdown = []
    drawdown_value = 0

    for i in range(len(pct_ret)):
        pct_return = pct_ret[i]
        if pct_return >= 0:  # Non-negative return
            if drawdown_value < 0:
                drawdown_value += pct_return
                if drawdown_value >= 0:
                    drawdown_value = 0
            else:
                drawdown_value = 0
        else:  # Negative return
            drawdown_value += pct_return
        Pfolio_Total_Daily_Drawdown.append(drawdown_value)

    return Pfolio_Total_Daily_Drawdown

def pfolio_max_drawdown(daily_drawdown):
    Pfolio_Total_Max_Drawdown = []
    prev_drawdown = daily_drawdown[0]
    drawdown = prev_drawdown
    for i in range(len(daily_drawdown)):
        curr_drawdown = daily_drawdown[i]
        if curr_drawdown < prev_drawdown:
            drawdown = curr_drawdown
            Pfolio_Total_Max_Drawdown.append(drawdown)
            prev_drawdown = curr_drawdown
        else:
            Pfolio_Total_Max_Drawdown.append(drawdown)
    
    return Pfolio_Total_Max_Drawdown

def get_trade_stats(pct_ret):
    total_trades = len(pct_ret)
    profitable_trades = 0
    losing_trades = 0
    for i in range (total_trades):
        if pct_ret[i] >= 0:
            profitable_trades += 1
        else:
            losing_trades += 1
    
    return total_trades,profitable_trades,losing_trades

def get_risk_reward(returns):
    Total_Possible_Loss = 0
    Total_Possible_Profit = 0
    for i in range (len(returns)):
        if returns[i] >= 0:
            Total_Possible_Profit += returns[i]
        else:
            Total_Possible_Loss += returns[i]
    return Total_Possible_Profit/abs(Total_Possible_Loss)

def returns_per_trade_pct(pct_returns):
    total_return_pct = 0 
    profit_return_pct = 0
    loss_return_pct = 0

    for i in range (len(pct_returns)):
        if pct_returns[i] >= 0:
            total_return_pct += pct_returns[i]
            profit_return_pct += pct_returns[i]
        else:
            total_return_pct += pct_returns[i]
            loss_return_pct += pct_returns[i]
    
    return total_return_pct, profit_return_pct, loss_return_pct

def main():
    Initial_capital = 1000000
    expiry_week = 'I'
    time = '10:29:59'
    entry_time  = '10:30'
    wing_percentage = 0.02
    stop_loss_percentage = 0.3
    target_percentage = 0.8
    dataframe = load_data('BANKNIFTY_2017_OPTIONS.csv')
    (CE_data,PE_data) = straddle_CE_and_PE(dataframe,expiry_week,time)
    CE_data.reset_index(drop=True,inplace=True)
    PE_data.reset_index(drop=True,inplace=True)

    futures_data = load_data('BANKNIFTY_2017_FUTURES.csv')
    futures_data_for_time = futures_data[futures_data['Time'] == time]
    futures_data_within_expiry = futures_data_for_time[futures_data_for_time['Contract'] == 'I']
    futures_price = fetch_futures_price_by_date(futures_data_within_expiry)

    selected_atm_call_data, selected_atm_put_data = (find_atm_call_put(futures_price,CE_data,PE_data))

    selected_atm_call_data.reset_index(drop=True,inplace=True)
    selected_atm_put_data.reset_index(drop=True,inplace=True)

    selected_wing_call_data,selected_wing_put_data = find_wing_call_put(futures_price,CE_data,PE_data,wing_percentage)

    selected_wing_call_data.reset_index(drop=True,inplace=True)
    selected_wing_put_data.reset_index(drop=True,inplace=True) 

    stop_loss_dict = {}
    target_dict = {}

    TradeReport = pd.DataFrame()
    TradeReport['Date'] = futures_price['Date']
    TradeReport['Expiry'] = selected_atm_call_data['Expiry'] + ' 0:00'
    TradeReport['En_Date'] = TradeReport['Date'] + f' {entry_time}'
    TradeReport['Ce_En_Date'] = TradeReport['Date'] + f' {entry_time}'
    TradeReport['Pe_En_Date'] = TradeReport['Date'] + f' {entry_time}'
    TradeReport['Fut_En_Price'] = futures_price['Close']
    TradeReport['Atm_strike'] = selected_atm_call_data['Strike']
    TradeReport['Ce_Short_strike'] = selected_atm_call_data['Strike']
    TradeReport['Pe_Short_strike'] = selected_atm_put_data['Strike']
    TradeReport['Ce_Long_strike'] = selected_wing_call_data['Strike']
    TradeReport['Pe_Long_strike'] = selected_wing_put_data['Strike']
    TradeReport['Ce_Short_En_Price'] = selected_atm_call_data['Close']
    TradeReport['Pe_Short_En_Price'] = selected_atm_put_data['Close']
    TradeReport['Ce_Long_En_Price'] = selected_wing_call_data['Close']
    TradeReport['Pe_Long_En_Price'] = selected_wing_put_data['Close']
    
    initial_prem_for_cal = []
    initial_premium = []
    for _, row in TradeReport.iterrows():
        entry_date = row['Date']
        atm_call_premium = row['Ce_Short_En_Price']
        atm_put_premium = row['Pe_Short_En_Price']
        wing_call_premium = row['Ce_Long_En_Price']
        wing_put_premium = row['Pe_Long_En_Price']

        total_premium = atm_call_premium+atm_put_premium-wing_call_premium-wing_put_premium
        initial_prem_for_cal.append(total_premium)
        total_premium_round = np.round(total_premium,0)
        initial_premium.append(total_premium_round)

        stop_loss,target = calculate_stop_loss_target(atm_call_premium,atm_put_premium,wing_call_premium,wing_put_premium,stop_loss_percentage,target_percentage)
        
        stop_loss_dict[entry_date] = stop_loss
        target_dict[entry_date] = target
    
    CE_monitor_data , PE_monitor_data = get_CE_and_PE(dataframe,expiry_week)

    trade_dict_atm_CE = (monitor_call_straddle_data(CE_monitor_data,selected_atm_call_data))
    trade_dict_atm_PE = (monitor_put_straddle_data(PE_monitor_data,selected_atm_put_data))
    trade_dict_wing_CE = (monitor_call_straddle_data(CE_monitor_data,selected_wing_call_data))
    trade_dict_wing_PE = (monitor_call_straddle_data(PE_monitor_data,selected_wing_put_data))

    exit_data = (monitor_premium(trade_dict_atm_CE,trade_dict_atm_PE,trade_dict_wing_CE,trade_dict_wing_PE,stop_loss_dict,target_dict))

    exit_data_df = pd.DataFrame(exit_data)
    exit_prem_for_cal = exit_data_df['Exit_Premium']
    exit_data_df['Exit_Premium'] = np.round(exit_data_df['Exit_Premium'],0)

    TradeReport.insert(loc=TradeReport.columns.get_loc('Pe_En_Date') + 1, column='Ce_Ex_Date', value=exit_data_df['Ce_Ex_Date'])
    TradeReport.insert(loc=TradeReport.columns.get_loc('Ce_Ex_Date') + 1, column='Pe_Ex_Date', value=exit_data_df['Pe_Ex_Date'])

    TradeReport.insert(loc=TradeReport.columns.get_loc('Ce_Short_En_Price') + 1, column='Ce_Short_Ex_Price', value=exit_data_df['Ce_Short_Ex_Price'])
    TradeReport.insert(loc=TradeReport.columns.get_loc('Pe_Short_En_Price') + 1, column='Pe_Short_Ex_Price', value=exit_data_df['Pe_Short_Ex_Price'])

    TradeReport.insert(loc=TradeReport.columns.get_loc('Ce_Long_En_Price') + 1, column='Ce_Long_Ex_Price', value=exit_data_df['Ce_Long_Ex_Price'])
    TradeReport.insert(loc=TradeReport.columns.get_loc('Pe_Long_En_Price') + 1, column='Pe_Long_Ex_Price', value=exit_data_df['Pe_Long_Ex_Price'])

    TradeReport['Initial_Premium'] = initial_premium
    TradeReport['Exit_Premium'] = exit_data_df['Exit_Premium']


    abs_returns = []

    for exit, initial in zip(exit_prem_for_cal,initial_prem_for_cal):
        abs_return = (initial-exit)
        abs_returns.append(abs_return)
    
    TradeReport['Returns_Abs'] = np.round(abs_returns,0)
    TradeReport['Quantity'] = 1
    TradeReport['Lot_Size'] = 100

    total_abs_returns = [100 * x for x in abs_returns]

    TradeReport['Total_Abs_Return'] = np.round(total_abs_returns,2)

    TradeReport['Total_Capital_Return'] = Initial_capital + TradeReport['Total_Abs_Return'].cumsum()

    pct_return = [x / 10000 for x in total_abs_returns]
    TradeReport['Total_Pct_Return'] = np.round(pct_return,4)

    Pfolio_Total_Daily_Drawdown = pfolio_daily_drawdown(pct_return)

    TradeReport['Pfolio_Total_Daily_Drawdown'] = np.round(Pfolio_Total_Daily_Drawdown,4)

    Pfolio_Max_Daily_Drawdown = pfolio_max_drawdown(Pfolio_Total_Daily_Drawdown)

    TradeReport['Pfolio_Max_Daily_Drawdown'] = np.round(Pfolio_Max_Daily_Drawdown,4)

    TradeReport.to_csv('TradeReport.csv',index=False)

    total_trades, profitable_trades, losing_trades = get_trade_stats(pct_return)
    hit_ratio = (profitable_trades/total_trades) * 100

    risk_reward_ratio = get_risk_reward(total_abs_returns)

    ending_value = total_abs_returns[-1] + Initial_capital
    annualized_returns = (ending_value/Initial_capital) - 1
    max_dd = (Pfolio_Max_Daily_Drawdown[-1])

    calmer_ratio = np.round(annualized_returns/abs(max_dd),2)

    total_return_pct, profit_return_pct, loss_return_pct = returns_per_trade_pct(pct_return)

    avg_return_per_trade, avg_profit_per_trade, avg_loss_per_trade = ((total_return_pct/total_trades),\
                                                                      (profit_return_pct/profitable_trades),\
                                                                      (loss_return_pct/losing_trades))
    
    max_profit_pct = max(pct_return)
    max_loss_pct = min(pct_return)

    data = [
    ['Capital', '', Initial_capital],
    ['Total Trades', '', total_trades],
    ['Profitable Trades', '', profitable_trades],
    ['Losing Trades', '', losing_trades],
    ['Hit Ratio', f'{hit_ratio:.2f}%', ''],
    ['Risk Reward', f'1:{risk_reward_ratio:.1f}', ''],
    ['Calmer Ratio', f'{calmer_ratio}', ''],
    ['Avg Return Per Trade', f'{avg_return_per_trade:.2f}%', f'{np.round(avg_return_per_trade * 10000,0)}'],
    ['Avg Profit Per Trade', f'{avg_profit_per_trade:.2f}%', f'{np.round(avg_profit_per_trade * 10000,0)}'],
    ['Avg Loss Per Trade', f'{avg_loss_per_trade:.2f}%', f'{np.round(avg_loss_per_trade * 10000,2)}'],
    ['Max Profit', f'{max_profit_pct:.2f}%', f'{np.round(max_profit_pct * 10000,2)}'],
    ['Max Loss', f'{max_loss_pct:.2f}%', f'{np.round(max_loss_pct * 10000,2)}'],
    ['Total Returns', f'{(total_return_pct):.2f}%', f'{np.round((total_return_pct) * 10000,2)}'],
    ['Max Drawdown', f'{(max_dd*100):.2f}%', f'{np.round(max_dd * 1000000,2)}']
    ]
    stats_df = pd.DataFrame(data, columns=['Stats', 'Percentage Return', 'Absolute Return'])
    stats_df.to_csv('PerformanceStats.csv',index = False)

if __name__ == '__main__':
    main()
    

