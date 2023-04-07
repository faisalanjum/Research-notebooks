
import os
import re
import time
import pandas as pd
import numpy as np
from os import walk
import collections

# Path of all files in folder and subfolder - CONTINUE Folder
def get_path_continue_files():
    List_of_files_continue = []
    folder_continue = "C:/source/2x4-data/app/data/Barchart/DailyDataDumpCME&NYMEX/dailyContinue/"
    folder_names_continue = [os.path.join(folder_continue, name) for name in os.listdir(folder_continue) if os.path.isdir(os.path.join(folder_continue, name))]
    folder_names_continue.append(folder_continue)
    for fold in folder_names_continue:
        filenames = next(walk(fold), (None, None, []))[2]  # [] if no file
        for file in filenames:
            List_of_files_continue.append(str(fold+"/"+file).replace('//','/'))
    List_of_files_continue = [val for val in List_of_files_continue if not val.endswith(".ipynb")] # remove all files end in .ipynb
    List_of_files_continue = [val for val in List_of_files_continue if not val.endswith(".pkl")] # remove all files end in .pkl
    List_of_files_continue = [val for val in List_of_files_continue if not val.endswith("expiry_df_other_symbols_Continue.csv")]
    List_of_files_continue = [val for val in List_of_files_continue if not val.endswith("expiry_df_Continue.csv")]
    return List_of_files_continue


# Path of all files in folder and subfolder - NEAREST Folder
def get_path_nearest_files():
    List_of_files_nearest = []
    folder_nearest = "C:/source/2x4-data/app/data/Barchart/DailyDataDumpCME&NYMEX/"
    folder_names_nearest = [os.path.join(folder_nearest, name) for name in os.listdir(folder_nearest) if os.path.isdir(os.path.join(folder_nearest, name))]
    folder_names_nearest.append(folder_nearest)
    folder_names_nearest.remove("C:/source/2x4-data/app/data/Barchart/DailyDataDumpCME&NYMEX/dailyContinue")
    folder_names_nearest.remove("C:/source/2x4-data/app/data/Barchart/DailyDataDumpCME&NYMEX/Archived CSV Files")
    for fold in folder_names_nearest:
        filenames = next(walk(fold), (None, None, []))[2]  # [] if no file
        for file in filenames:
            List_of_files_nearest.append(str(fold+"/"+file).replace('//','/'))
    List_of_files_nearest = [val for val in List_of_files_nearest if not val.endswith(".ipynb")] # remove all files end in .ipynb
    List_of_files_nearest = [val for val in List_of_files_nearest if not val.endswith(".pkl")] # remove all files end in .pkl
    List_of_files_nearest = [val for val in List_of_files_nearest if not val.endswith("combined_dataframe.csv")]
    List_of_files_nearest = [val for val in List_of_files_nearest if not val.endswith("expiry_df.csv")]
    List_of_files_nearest = [val for val in List_of_files_nearest if not val.endswith("expiry_df_other_symbols.csv")]
    return List_of_files_nearest

#Combined file names from both folders - Continue & Nearest and gives back a dictionary with key of Nearest and Value of Continue
# Matching Nearest & Continue FILES - removes "_Continue" at the end but before ".csv"
# For each file in Nearest, changes the string of the path to make it similar to one in Continue and the check if its exits in Continue
def combine_paths_nearest_continue(List_of_files_nearest, List_of_files_continue):
    file_paths_dict = {}
    for file_nearest in List_of_files_nearest:
        continue_string = file_nearest[file_nearest.rindex('/')+1:]
        continue_string = str(continue_string.replace(".csv","")+"_Continue.csv")
        idx_continue = [i for i, s in enumerate(List_of_files_continue) if continue_string in s]
        file_paths_dict[file_nearest] = List_of_files_continue[idx_continue[0]]
    return file_paths_dict

# Get Expiry dataframe from csv
def get_expiry():
    expiry_nearest = pd.read_csv("C:/source/2x4-data/app/data/Barchart/DailyDataDumpCME&NYMEX/expiry_df.csv", index_col=0)
    return expiry_nearest


# Collect all contracts for all commodities in a dictionary and a list of dates
def all_contracts_dates_all_commodities(file_paths_dict,expiry_nearest):
    file_path_cmdty_name = {}
    all_dates = []
    cmdty_ctrct_dict = {}
    cmdty_no_data = []
    temp_list = []
    cmdty_no_expiry = []
    cmdty_ctrct_expiry_dict = collections.defaultdict(dict) # If the amount of nesting you need is fixed, use collections.defaultdict.

    for file in file_paths_dict.keys():
        cmdty_name = file[file.rindex('/')+1:].replace('_daily_price.csv','')
        
        try:
            list_nearest_ctrct = pd.read_csv(file)['symbol'].unique().tolist()
            list_nearest_dates = pd.read_csv(file)['tradingDay'].unique().tolist()

        except:
            cmdty_no_data.append(cmdty_name)
            temp_list.append(file_paths_dict[file])

        try:
            list_continue_ctrct = pd.read_csv(file_paths_dict[file],index_col=0)['symbol'].unique().tolist()
            list_continue_dates = pd.read_csv(file_paths_dict[file],index_col=0)['tradingDay'].unique().tolist()

        except:
            cmdty_no_data.append(cmdty_name)
            temp_list.append(file_paths_dict[file])

        cmdty_ctrct_dict[cmdty_name] = set(list_nearest_ctrct + list_continue_ctrct)
        file_path_cmdty_name[file] = cmdty_name

        all_dates = all_dates + list_nearest_dates + list_continue_dates

        for ctrcts in cmdty_ctrct_dict[cmdty_name]:
            try:
                cmdty_ctrct_expiry_dict[cmdty_name][ctrcts] = expiry_nearest.loc[ctrcts,'expirationDate'], expiry_nearest.loc[ctrcts,'exchange'], expiry_nearest.loc[ctrcts,'contract'].replace('_','')
            except:
                cmdty_no_expiry.append(ctrcts)

            # To do dict of dicts
        
    temp_list =  set(temp_list)
    all_dates = set(all_dates)
    cmdty_no_data = set(cmdty_no_data)
    all_dates = pd.Series(pd.DatetimeIndex(all_dates))
    all_dates =  all_dates.sort_values(ascending=False).dt.strftime('%Y-%m-%d').tolist()  

    return cmdty_ctrct_dict, all_dates, cmdty_no_data, cmdty_no_expiry, cmdty_ctrct_expiry_dict, file_path_cmdty_name 

# This is the one that gives you all data points for a commodity 
# Combine dataframes from continue and nearest for one commodity - note file_path is the key in file_paths_dict and has to be a nearest file
def combine_data_continue_nearest(file_path, file_paths_dict):
    combined = pd.concat([pd.read_csv(file_path,index_col=0), pd.read_csv(file_paths_dict[file_path],index_col=0)])
    combined.drop_duplicates(inplace=True)
    return combined

# Get an expiry dataframe for all the contracts in a commodity (cmdty_name). Note cmdty_name is the key in cmdty_ctrct_expiry_dict
def get_one_commodity_ctrcts_expiry(cmdty_ctrct_expiry_dict, cmdty_name):
    one_commodity_expiry = pd.DataFrame(cmdty_ctrct_expiry_dict[cmdty_name]).T
    one_commodity_expiry.columns = ['expiry','exchange','contract']
    one_commodity_expiry.expiry = pd.to_datetime(one_commodity_expiry.expiry)
    one_commodity_expiry.sort_values(by=['expiry'], ascending= False, inplace=True)
    return one_commodity_expiry

# This lines up values for based on expiry date for each contract within a commodity along with their expiry dates in the first row

def get_ohlcv_by_contract_date(combined,one_commodity_expiry):
    combined_pivot = {}
    new_combined_pivot = {}
    for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
        combined_pivot[ohlcv_opt] = pd.pivot_table(combined, values=ohlcv_opt, index='tradingDay', columns='symbol').sort_index(ascending=False)
        combined_pivot[ohlcv_opt].reindex(one_commodity_expiry.index, axis=1)
        new_combined_pivot[ohlcv_opt] = pd.concat([pd.DataFrame(index = one_commodity_expiry.index, data=list(one_commodity_expiry['expiry']),columns=['Expiry']).T,combined_pivot[ohlcv_opt]], axis=0)
    return new_combined_pivot


# For each date it tells you which was the active contract (also gives a column for any OHLCV option you choose for that date for the particular contract). 
# You can choose active contract in NearestMonth. Default is 1st month contract # starts from month 1 
def get_active_contract_by_date(new_combined_pivot, NearestMonth = 1):
    # Takes a lot of Time - ~ 2minutes for 1 commodity because OF nsmallest(NearestMonth), sort_values is faster ~48 seconds
    NearestContract_df = pd.DataFrame()
    
    # It doesn't matter if we use 'close' instead of OHLCV below since we just need to find the nearest contract
    for dt in new_combined_pivot['close'].index[1:]:
        comp = pd.to_datetime(new_combined_pivot['close'].loc['Expiry'])[dt <= pd.to_datetime(new_combined_pivot['close'].loc['Expiry'])] - pd.to_datetime(dt)
        # active = comp.nsmallest(NearestMonth).index[NearestMonth-1]
        active = comp.sort_values().head(NearestMonth).index[NearestMonth-1]
        temp3 = pd.DataFrame(data=[[dt,active]], columns=['dt','active'])
        NearestContract_df = pd.concat([NearestContract_df,temp3])

    # Taking 40 seconds
    for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
        NearestContract_df[ohlcv_opt] = np.nan
        NearestContract_df.reset_index(inplace = True)
        NearestContract_df.drop('index', axis=1,inplace = True)
        for i in range(len(NearestContract_df)):
            NearestContract_df.loc[i, ohlcv_opt] = new_combined_pivot[ohlcv_opt].loc[NearestContract_df.iloc[i][0],NearestContract_df.iloc[i][1]]
    
    return NearestContract_df


# Setup df to compare the difference and ratio methodlogies of the Panama Adjustment methods
# backAdjustDays = 0       # 0 means expired on the day of expiry
def get_comparison_df(NearestContract_df,new_combined_pivot, backAdjustDays = 0):
    NearestContractShifted_df = NearestContract_df.copy(deep=True)
    shifted_contracts = pd.Series(NearestContractShifted_df.active[0:backAdjustDays+1].values)
    NearestContractShifted_df.active = NearestContractShifted_df.active.shift(periods=(backAdjustDays+1))
    NearestContractShifted_df.active.iloc[0:backAdjustDays+1] = shifted_contracts
    NearestContractShifted_df.active.reset_index(drop=True, inplace=True)

    for i in range(len(NearestContractShifted_df)):
        for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
            NearestContractShifted_df.loc[i, ohlcv_opt] = new_combined_pivot[ohlcv_opt].loc[NearestContractShifted_df.iloc[i][0],NearestContractShifted_df.iloc[i][1]]
    
    NearestContract_df.index = NearestContract_df.dt
    NearestContract_df.drop('dt', axis=1,inplace = True)
    NearestContractShifted_df.index = NearestContractShifted_df.dt
    NearestContractShifted_df.drop('dt', axis=1,inplace = True)
    # ComparisonNearestContractdf=NearestContract_df.compare(NearestContractShifted_df)
    ComparisonNearestContractdf=compare_two_dfs(NearestContract_df,NearestContractShifted_df)

    for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
        ComparisonNearestContractdf[str('Diff_'+ohlcv_opt)] = ComparisonNearestContractdf.loc[:,(ohlcv_opt,'other')].sub(ComparisonNearestContractdf.loc[:,(ohlcv_opt,'self')], axis = 0)
        ComparisonNearestContractdf[str('Prod_'+ohlcv_opt)] = ComparisonNearestContractdf.loc[:,(ohlcv_opt,'other')].div(ComparisonNearestContractdf.loc[:,(ohlcv_opt,'self')])

    return ComparisonNearestContractdf


def compare_two_dfs(NearestContract_df,NearestContractShifted_df):
    idx_to_compare = NearestContract_df.active.compare(NearestContractShifted_df.active).index

    compare_columns = []
    for t in NearestContract_df.columns:
        compare_columns.append(tuple([t,'self']))
        compare_columns.append(tuple([t,'other']))

    ComparisonNearestContractdf = pd.DataFrame(index=idx_to_compare, columns=pd.MultiIndex.from_tuples(compare_columns))

    for idx in idx_to_compare:
        for t in NearestContract_df.columns:
            ComparisonNearestContractdf.loc[idx,tuple([t,'self'])] = NearestContract_df.loc[idx,t]
            ComparisonNearestContractdf.loc[idx,tuple([t,'other'])] = NearestContractShifted_df.loc[idx,t]

    return ComparisonNearestContractdf



def compute_adjusted_values(NearestContract_df, ComparisonNearestContractdf, backAdjustDays, method = 'absolute', weights = None, weighted_average = False):
    
    # remove these if else stataements by combing methods for both absolute and relative
    cumulative_df = {}
    final_df = {}
    ohlcv_DF = pd.DataFrame()

    if weights is None:
            # weights = [1/(backAdjustDays+1)]*(backAdjustDays+1) # pass default weights
            weights = [0]* backAdjustDays + [1]*1 # For 100% weight on the last day
    
    if(method == 'absolute'):
        
        # Note that for all days from expiry to backAdjustDays, the multiplier remains the same. It also remains same from backadjustDays till next expiry
        
        for ohlcv_opt in ['close','open','high','low']:

            column_name = str('Diff_'+ohlcv_opt)
            
            # Note that for all days from expiry to backAdjustDays, the multiplier is based on that particular day. From backAdjustDays onwards till next expiry, the multiplier is based on weighted average
            cumulative_df[ohlcv_opt] = pd.DataFrame(index = ComparisonNearestContractdf.index, data = ComparisonNearestContractdf[column_name])
            cumulative_df[ohlcv_opt].reset_index(inplace=True)

            # Put weighted average for all days from expiry till backAdjustDays
            cumulative_df[ohlcv_opt]['wt_avg'] = cumulative_df[ohlcv_opt][column_name].groupby(cumulative_df[ohlcv_opt].index//(backAdjustDays+1)).apply(lambda x: np.multiply(x,weights)).groupby(cumulative_df[ohlcv_opt].index//(backAdjustDays+1)).transform('sum')
            cumulative_df[ohlcv_opt]['Cum_effect'] = cumulative_df[ohlcv_opt]['wt_avg']

            # flag == 1 represents (expiry + backAdjustDays)
            pivot_backAdjustDays_idx = [x for x in cumulative_df[ohlcv_opt].index if (x % (backAdjustDays+1)) == backAdjustDays]

            cumulative_df[ohlcv_opt]['flag'] = [1 if x in pivot_backAdjustDays_idx else 0 for x in cumulative_df[ohlcv_opt].index]

            cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Cum_effect'] = 0 # Put 0 for all days before backAdjustDays so as to make cumsum work
            cumulative_df[ohlcv_opt]['Cum_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'].cumsum()

            # Once we have forward filled the cumulative effect, we can take the cumulative effect for days before backAdjustDays
            # weighted_average - True will apply weighted average to data from expiry till backAdjustDays - False will apply that particular day's either relative difference or relative ratio (to cumulative difference/ratio)
            if(weighted_average):
                #The following Line adds weighted effect to days before backAdjustDays
                cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'] + cumulative_df[ohlcv_opt]['wt_avg']
            else:
                #The following Line adds daily differece effect to days before backAdjustDays
                cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'] + cumulative_df[ohlcv_opt][str('Diff_'+ohlcv_opt)]

            cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 1, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect']

            cumulative_df[ohlcv_opt].index  = cumulative_df[ohlcv_opt].dt
            cumulative_df[ohlcv_opt].drop('dt', axis=1,inplace = True)
            
            # Using Total effect of only 'close' column for all OHCL since if we don't use Total effect of 'close' column, we will get issues like "adj_low" being higher than "adj_high" for some dates
            final_df[ohlcv_opt] = pd.DataFrame(index = NearestContract_df.index, data = cumulative_df['close']['Total_effect'] )
            final_df[ohlcv_opt].fillna(method='ffill',inplace = True)
            
            # Use either of 2 lines to fill NaN values with 0 
            final_df[ohlcv_opt].iloc[np.argwhere(final_df[ohlcv_opt].index > cumulative_df[ohlcv_opt].index[0])] = 0 # Add 0 to Total Effect prior to first expiry date
            # final_df[ohlcv_opt].fillna(0,inplace = True)
            
            final_df[ohlcv_opt][ohlcv_opt] = NearestContract_df[ohlcv_opt].values
            final_df[ohlcv_opt][str('Adj_'+ohlcv_opt)] = final_df[ohlcv_opt][ohlcv_opt] + final_df['close']['Total_effect']
            ohlcv_DF = pd.concat([ohlcv_DF,final_df[ohlcv_opt]],axis=1)

        ohlcv_DF = pd.concat([ohlcv_DF,NearestContract_df[['volume','openInterest']]],axis=1)
        ohlcv_DF.drop(['Total_effect'],axis=1,inplace=True)


        return ohlcv_DF

    elif(method == 'relative'):

        # If you Total Effect equals Zero, then it wasn't able to find values for both simultaneous contracts for the date in question. So the net impact is no adjustment
        for ohlcv_opt in ['close','open','high','low']:

            # CHANGE 1
            column_name = str('Prod_'+ohlcv_opt)
            
            # Note that for all days from expiry to backAdjustDays, the multiplier is based on that particular day. From backAdjustDays onwards till next expiry, the multiplier is based on weighted average
            cumulative_df[ohlcv_opt] = pd.DataFrame(index = ComparisonNearestContractdf.index, data = ComparisonNearestContractdf[column_name])
            cumulative_df[ohlcv_opt].reset_index(inplace=True)

            # Put weighted average for all days from expiry till backAdjustDays -
            cumulative_df[ohlcv_opt]['wt_avg'] = cumulative_df[ohlcv_opt][column_name].groupby(cumulative_df[ohlcv_opt].index//(backAdjustDays+1)).apply(lambda x: np.multiply(x,weights)).groupby(cumulative_df[ohlcv_opt].index//(backAdjustDays+1)).transform('sum') 
            cumulative_df[ohlcv_opt]['Cum_effect'] = cumulative_df[ohlcv_opt]['wt_avg']

            # flag == 1 represents the day on (expiry + backAdjustDays)
            pivot_backAdjustDays_idx = [x for x in cumulative_df[ohlcv_opt].index if (x % (backAdjustDays+1)) == backAdjustDays]
            cumulative_df[ohlcv_opt]['flag'] = [1 if x in pivot_backAdjustDays_idx else 0 for x in cumulative_df[ohlcv_opt].index]

            #CHANGE 2
            if(backAdjustDays == 0): # Note since if backAdjustDays = 0, all flags will be 1 and thus they need to be made equal to 1
                pass # If the backAdjustDays equals 0, then its already 1 so no need to put it as 1 
            else:
                cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Cum_effect'] = 1 # Put 1 for all days before backAdjustDays so as to make cumprod work

            cumulative_df[ohlcv_opt]['Cum_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'].cumprod()

            # Once we have forward filled the cumulative effect, we can take the cumulative effect for days before backAdjustDays
            
            
            # Change 4
            if(backAdjustDays == 0):
                cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 1, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect']
            else:
                # weighted_average - True will apply weighted average to data from expiry till backAdjustDays - False will apply that particular day's either relative difference or relative ratio (to cumulative difference/ratio)
                if(weighted_average):
                    #The following Line adds weighted effect to days before backAdjustDays
                    cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'] * cumulative_df[ohlcv_opt]['wt_avg']
                else:
                    #The following Line adds daily product effect to days before backAdjustDays
                    cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'] * cumulative_df[ohlcv_opt][str('Prod_'+ohlcv_opt)]    

                cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 1, 'Total_effect'] = cumulative_df[ohlcv_opt]['Cum_effect']
            
            
            cumulative_df[ohlcv_opt].index  = cumulative_df[ohlcv_opt].dt
            cumulative_df[ohlcv_opt].drop('dt', axis=1,inplace = True)

            # Using Total effect of only 'close' column for all OHCL since if we don't use Total effect of 'close' column, we will get issues like "adj_low" being higher than "adj_high" for some dates
            final_df[ohlcv_opt] = pd.DataFrame(index = NearestContract_df.index, data = cumulative_df['close']['Total_effect'])
            
        
            # slight change needs to be made, instead of changing everycolumn - only change 'Total_effect' column
            final_df[ohlcv_opt].fillna(method='ffill',inplace = True)
            final_df[ohlcv_opt].iloc[np.argwhere(final_df[ohlcv_opt].index > cumulative_df[ohlcv_opt].index[0])] = 1 # Add 1 to Total Effect prior to first expiry date
            

            # When we get a value of 0 for Total_effect, it means that we were unable to find values for both simultaneous contracts for the date in question. So we just use the same last effect backward in time till we get a 
            final_df[ohlcv_opt]['Total_effect'].replace(to_replace=0, method='ffill', inplace=True)

            final_df[ohlcv_opt][ohlcv_opt] = NearestContract_df[ohlcv_opt].values

            # Change 5
            final_df[ohlcv_opt][str('Adj_'+ohlcv_opt)] = final_df[ohlcv_opt][ohlcv_opt] * final_df['close']['Total_effect']
            
            ohlcv_DF = pd.concat([ohlcv_DF,final_df[ohlcv_opt]],axis=1)

        ohlcv_DF = pd.concat([ohlcv_DF,NearestContract_df[['volume','openInterest']]],axis=1)
        ohlcv_DF.drop(['Total_effect'],axis=1,inplace=True)

        return ohlcv_DF

    else:
        print('Method not supported')   



def save_csv_file(path, cmdty_name, ohlcv_DF):
    file_path = path + "/" + cmdty_name + '.csv'
    print("Saving file: ", file_path)
    ohlcv_DF.to_csv(file_path)

def get_adjusted_data(path,file_path,file_paths_dict,cmdty_ctrct_expiry_dict, cmdty_name, NearestMonth, backAdjustDays,method, weights,weighted_average):

    start = time. time()
    print("Computing",cmdty_name)

    combined = combine_data_continue_nearest(file_path, file_paths_dict)
    one_commodity_expiry = get_one_commodity_ctrcts_expiry(cmdty_ctrct_expiry_dict, cmdty_name)
    new_combined_pivot = get_ohlcv_by_contract_date(combined,one_commodity_expiry)

    # Takes about 50 seconds
    NearestContract_df = get_active_contract_by_date(new_combined_pivot, NearestMonth)
    Second_NearestContract_df = get_active_contract_by_date(new_combined_pivot, 2)
    ComparisonNearestContractdf = get_comparison_df(NearestContract_df,new_combined_pivot, backAdjustDays)
    ohlcv_DF = compute_adjusted_values(NearestContract_df, ComparisonNearestContractdf, backAdjustDays, method, weights, weighted_average)
    
    # Uncomment when ready to save
    # save_csv_file(path, cmdty_name, ohlcv_DF)

    end = time. time()
    print("Total Time Taken to save -> ",cmdty_name,": ", round(end - start,0)," seconds")
    
    return ohlcv_DF

def get_switch_data(NearestContract_df,expiry_nearest,Second_NearestContract_df,compare_criteria = 'openInterest',cut_off = 2):
        # Pass NearestContract_df, expiry_nearest, and Second_NearestContract_df to the function

    #Choices
    # compare_criteria = 'openInterest' #'openInterest' or 'volume'
    # cut_off = 2 - so when Next month OpenInterest or Volume is twice more than First Month

    Switch_df = NearestContract_df.copy(deep=True)

    Switch_df['compare_ratio'] =  Second_NearestContract_df[compare_criteria]/NearestContract_df[compare_criteria]
    Switch_df.drop(['open','high','low','close','volume','openInterest'], axis=1,inplace = True)
    # Switch_df['Second_active'] = Second_NearestContract_df['active'] # Put Next Contract as a column

    # Remove where openInterest/volume for either NearestContract_df or Second_NearestContract_df is zero or NA
    Switch_df = Switch_df[(Switch_df['compare_ratio'].notnull()) & (Switch_df['compare_ratio'] >= cut_off)]
    Switch_df = Switch_df.loc[Switch_df.groupby('active').compare_ratio.idxmin()] # Find the row with the first instance where our criteria meets i.e. where cut_off > compare_ratio

    # Can also use dt instead of Switch_dt
    # Switch_df.rename(columns={'dt':'Switch_dt'}, inplace=True)

    Switch_df['Switch_dt'] = np.nan
    Switch_df['expiry_dt'] = np.nan
    Switch_df['days_to_expiry'] = np.nan

    for ctrcts in Switch_df['active'].unique():
        try:
            Switch_df.loc[Switch_df['active'] == ctrcts, 'Switch_dt'] = NearestContract_df.iloc[Switch_df[Switch_df['active'] == ctrcts].index[0]]['dt']
            Switch_df.loc[Switch_df['active'] == ctrcts, 'expiry_dt'] = pd.DatetimeIndex([expiry_nearest.loc[ctrcts]['expirationDate']]).strftime('%Y-%m-%d')
            Switch_df.loc[Switch_df['active'] == ctrcts, 'days_to_expiry'] = NearestContract_df[NearestContract_df['dt'] == Switch_df[Switch_df['active'] == ctrcts]['Switch_dt'].values[0]].index[0] - NearestContract_df[NearestContract_df['dt'] == Switch_df[Switch_df['active'] == ctrcts]['expiry_dt'].values[0]].index[0]
        except:
            pass

    Switch_df.dropna(inplace=True)

    #List of contracts missing in the Switch_df
    missing_contracts = list(set(NearestContract_df['active'].unique()) - set(Switch_df['active'].unique()))

    doe_mean = Switch_df['days_to_expiry'].mean().__floor__()

    ctrcts_still_missing = []

    for miss in missing_contracts:
        try:
            Switch_df = pd.concat([Switch_df, pd.DataFrame(data=[[miss,np.nan,np.nan,np.nan]], columns=['active','Switch_dt','expiry_dt','days_to_expiry'])])
            Switch_df.loc[Switch_df['active'] == miss, 'expiry_dt'] = pd.DatetimeIndex([expiry_nearest.loc[miss]['expirationDate']]).strftime('%Y-%m-%d')
            Switch_df.loc[Switch_df['active'] == miss, 'days_to_expiry'] = doe_mean
            Switch_df.loc[Switch_df['active'] == miss, 'Switch_dt'] = NearestContract_df.iloc[NearestContract_df[NearestContract_df['dt'] == Switch_df.loc[Switch_df['active'] == miss]['expiry_dt'][0]].index[0] + doe_mean]['dt']
        except:
            ctrcts_still_missing.append(miss) # Mostly these contracts are the front month contracts becuase NearestContractdf doesn't take them inot account

    first_second_ctrcts = pd.concat([NearestContract_df['active'], Second_NearestContract_df['active']],axis=1)
    first_second_ctrcts.columns = ['First','Second']

    # To put Second Month Contract in Dataframe
    for ctrcts in Switch_df['active'].unique():
        Switch_df.loc[Switch_df['active'] == ctrcts, 'Second_active'] = first_second_ctrcts[first_second_ctrcts['First'] == ctrcts]['Second'].unique()[0]
        
    Switch_df.drop(['dt','compare_ratio','days_to_expiry'], axis=1,inplace = True)


    # For the contracts which haven't got the expiry date in NearestContract_df (yet to be expired) OR JUST don't have the data
    try:
        LastCtrct_Switch = NearestContract_df.copy(deep=True)
        latest_ctrct = NearestContract_df['active'][0]
        LastCtrct_Switch = LastCtrct_Switch[LastCtrct_Switch['active'] == latest_ctrct]
        LastCtrct_Switch['compare_ratio'] =  Second_NearestContract_df[compare_criteria]/NearestContract_df[compare_criteria]
        LastCtrct_Switch = LastCtrct_Switch[(LastCtrct_Switch['compare_ratio'].notnull()) & (LastCtrct_Switch['compare_ratio'] >= cut_off)]
        LastCtrct_Switch = LastCtrct_Switch.loc[LastCtrct_Switch.groupby('active').compare_ratio.idxmin()] # Find the row with the first instance where our criteria meets i.e. where cut_off > compare_ratio
        Switch_df.loc[Switch_df['active'] == latest_ctrct, 'Switch_dt']  = LastCtrct_Switch['dt'].values[0]

    except:
        pass

    return Switch_df


def get_comparison_df_switch(NearestContract_df,new_combined_pivot,Switch_df):
    # Takes 40 seconds
    NearestContractShifted_df = NearestContract_df.copy(deep=True)

    for ctrcts in NearestContractShifted_df['active'].unique():
        NearestContractShifted_df.loc[(NearestContractShifted_df['dt'] >= Switch_df[Switch_df['active'] == ctrcts]['Switch_dt'].values[0]) & (NearestContractShifted_df['dt'] <= Switch_df[Switch_df['active'] == ctrcts]['expiry_dt'].values[0] ), 'active'] = Switch_df[Switch_df['active'] == ctrcts]['Second_active'].values[0]

    NearestContractShifted_df.active.reset_index(drop=True, inplace=True)

    for i in range(len(NearestContractShifted_df)):
        for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
            NearestContractShifted_df.loc[i, ohlcv_opt] = new_combined_pivot[ohlcv_opt].loc[NearestContractShifted_df.iloc[i][0],NearestContractShifted_df.iloc[i][1]]

    NearestContract_df.index = NearestContract_df.dt
    NearestContract_df.drop('dt', axis=1,inplace = True)

    NearestContractShifted_df.index = NearestContractShifted_df.dt
    NearestContractShifted_df.drop('dt', axis=1,inplace = True)
    idx_to_compare = NearestContract_df.active.compare(NearestContractShifted_df.active).index

    compare_columns = []
    for t in NearestContract_df.columns:
        compare_columns.append(tuple([t,'self']))
        compare_columns.append(tuple([t,'other']))

    ComparisonNearestContractdf = pd.DataFrame(index=idx_to_compare, columns=pd.MultiIndex.from_tuples(compare_columns))

    for idx in idx_to_compare:
        for t in NearestContract_df.columns:
            ComparisonNearestContractdf.loc[idx,tuple([t,'self'])] = NearestContract_df.loc[idx,t]
            ComparisonNearestContractdf.loc[idx,tuple([t,'other'])] = NearestContractShifted_df.loc[idx,t]

    for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
        ComparisonNearestContractdf[str('Diff_'+ohlcv_opt)] = ComparisonNearestContractdf.loc[:,(ohlcv_opt,'other')].sub(ComparisonNearestContractdf.loc[:,(ohlcv_opt,'self')], axis = 0)
        ComparisonNearestContractdf[str('Prod_'+ohlcv_opt)] = ComparisonNearestContractdf.loc[:,(ohlcv_opt,'other')].div(ComparisonNearestContractdf.loc[:,(ohlcv_opt,'self')], axis = 0)


    return ComparisonNearestContractdf



def compute_adjusted_values_switch(NearestContract_df, ComparisonNearestContractdf, Switch_df):
    
    cumulative_df = {}
    final_df = {}
    ohlcv_DF = pd.DataFrame()

    for ohlcv_opt in ['close','open','high','low']:

        # CHANGE 1
        column_name = str('Prod_'+ohlcv_opt)
        
        cumulative_df[ohlcv_opt] = pd.DataFrame(index = ComparisonNearestContractdf.index, data = ComparisonNearestContractdf[column_name])
        
        # Flag = 1 when switching occurs
        cumulative_df[ohlcv_opt]['flag'] = [1 if x in Switch_df['Switch_dt'].unique() else 0 for x in cumulative_df[ohlcv_opt].index]
        cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 1, 'Cum_effect'] = cumulative_df[ohlcv_opt][str('Prod_'+ohlcv_opt)]
        
        # Put 1 for all days Not on Switching days to make cumprod work
        cumulative_df[ohlcv_opt].loc[cumulative_df[ohlcv_opt]['flag'] == 0, 'Cum_effect'] = 1 
        cumulative_df[ohlcv_opt]['Cum_effect'] = cumulative_df[ohlcv_opt]['Cum_effect'].cumprod()

        # Using Total effect of only 'close' column for all OHCL since if we don't use Total effect of 'close' column, we will get issues like "adj_low" being higher than "adj_high" for some dates
        final_df[ohlcv_opt] = pd.DataFrame(index = NearestContract_df.index, data = cumulative_df['close']['Cum_effect'])
        
        # slight change needs to be made, instead of changing everycolumn - only change 'Cum_effect' column
        final_df[ohlcv_opt].fillna(method='ffill',inplace = True)
        final_df[ohlcv_opt].iloc[np.argwhere(final_df[ohlcv_opt].index > cumulative_df[ohlcv_opt].index[0])] = 1 # Add 1 to Cum Effect prior to first expiry date

        # When we get a value of 0 for Total_effect, it means that we were unable to find values for both simultaneous contracts for the date in question. So we just use the same last effect backward in time till we get a 
        final_df[ohlcv_opt]['Cum_effect'].replace(to_replace=0, method='ffill', inplace=True)
        final_df[ohlcv_opt][ohlcv_opt] = NearestContract_df[ohlcv_opt].values

        final_df[ohlcv_opt][str('Adj_'+ohlcv_opt)] = final_df[ohlcv_opt][ohlcv_opt] * final_df['close']['Cum_effect']    
        ohlcv_DF = pd.concat([ohlcv_DF,final_df[ohlcv_opt]],axis=1)

    ohlcv_DF = pd.concat([ohlcv_DF,NearestContract_df[['volume','openInterest']]],axis=1)
    ohlcv_DF.drop(['Cum_effect'],axis=1,inplace=True)

        
    return ohlcv_DF


def get_adjusted_data_switch(path,file_path,file_paths_dict,cmdty_ctrct_expiry_dict, cmdty_name, NearestMonth,expiry_nearest,compare_criteria,cut_off):
    
    start = time. time()
    print("Computing",cmdty_name)

    combined = combine_data_continue_nearest(file_path, file_paths_dict)
    one_commodity_expiry = get_one_commodity_ctrcts_expiry(cmdty_ctrct_expiry_dict, cmdty_name)
    new_combined_pivot = get_ohlcv_by_contract_date(combined,one_commodity_expiry)

    # Takes about 50 seconds
    NearestContract_df = get_active_contract_by_date(new_combined_pivot, NearestMonth)
    Second_NearestContract_df = get_active_contract_by_date(new_combined_pivot, 2)
    Switch_df = get_switch_data(NearestContract_df,expiry_nearest,Second_NearestContract_df,compare_criteria,cut_off)
    ComparisonNearestContractdf = get_comparison_df_switch(NearestContract_df,new_combined_pivot,Switch_df)
    ohlcv_DF = compute_adjusted_values_switch(NearestContract_df, ComparisonNearestContractdf, Switch_df)
    
    # Uncomment when ready to save
    save_csv_file(path, cmdty_name, ohlcv_DF)

    end = time. time()
    print("Total Time Taken to save -> ",cmdty_name,": ", round(end - start,0)," seconds")
    
    return ohlcv_DF

# This is a simple function that takes in a csv address for undjusted OHLCV and returns adjusted ohlcv based on the criteria selected for a contract to switch - no retrospective adjustment is made
def simple_adjustment(f, compare_criteria, cut_off):
    # Read all file paths and expiry dates
    
    List_of_files_continue = get_path_continue_files()
    List_of_files_nearest = get_path_nearest_files()
    file_paths_dict = combine_paths_nearest_continue(List_of_files_nearest,List_of_files_continue)
    expiry_nearest = get_expiry()
    combined = combine_data_continue_nearest(f, file_paths_dict)

    cmdty_ctrct_dict, all_dates, cmdty_no_data, cmdty_no_expiry, cmdty_ctrct_expiry_dict, file_path_cmdty_name  = all_contracts_dates_all_commodities(file_paths_dict,expiry_nearest)
    cmdty_name = file_path_cmdty_name[f]
    one_commodity_expiry = get_one_commodity_ctrcts_expiry(cmdty_ctrct_expiry_dict, cmdty_name)
    
    NearestMonth = 1 
    new_combined_pivot = get_ohlcv_by_contract_date(combined,one_commodity_expiry)          
    NearestContract_df = get_active_contract_by_date(new_combined_pivot, NearestMonth)          
    Second_NearestContract_df = get_active_contract_by_date(new_combined_pivot, 2) 
    Switch_df = get_switch_data(NearestContract_df,expiry_nearest,Second_NearestContract_df,compare_criteria,cut_off)    
    NearestContract_df.index = pd.DatetimeIndex(NearestContract_df['dt'])
    NearestContract_df.drop(['dt'], axis=1, inplace=True)
    Second_NearestContract_df.index = pd.DatetimeIndex(Second_NearestContract_df['dt'])
    Second_NearestContract_df.drop(['dt'], axis=1, inplace=True)
    combined_first_second_df = pd.concat([NearestContract_df.active,Second_NearestContract_df.active],axis=1).reindex(NearestContract_df.index)
    switch_dates = pd.DatetimeIndex(Switch_df['Switch_dt'].unique())
    combined_first_second_df['flag']= [1 if x in switch_dates else 0 for x in combined_first_second_df.index]
    combined_first_second_df.columns = ['first','second','flag']
    combined_first_second_df.loc[combined_first_second_df['flag'] == 1, 'Final'] = combined_first_second_df['second']
    combined_first_second_df.Final.fillna(method='bfill',inplace = True)
    combined_first_second_df.drop(['first','second','flag'],axis=1,inplace=True)
    combined_first_second_df.dropna(inplace=True)
    combined_first_second_df.index = pd.DatetimeIndex(combined_first_second_df.index).strftime('%Y-%m-%d')

    for dt in combined_first_second_df.index:
        for ohlcv_opt in ['open','high','low','close','volume','openInterest']:
                combined_first_second_df.loc[dt, ohlcv_opt] = new_combined_pivot[ohlcv_opt].loc[dt,combined_first_second_df.loc[dt,'Final']]

    return combined_first_second_df