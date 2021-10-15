### Import modules and packages
import numpy as np
import pandas as pd

import math
from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler ## for Feature Scaling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from pandas.tseries.offsets import CustomBusinessDay

# my modules
import mongodb_stock
from email_notice_function import sendEmailNotice


def myLSTM(raw_df, X_features, y_features, n_past_days, n_future_days, loss='mean_squared_error', 
lr=0.01, epochs=12, batch_size=32, ticker=None):

    """
    # LSTM ml model prediction of stock prices for <n_future_days> days using <n_past_days> days data
        - Args: raw_df, X_features, y_features, n_past_days, n_future_days, 
                loss, lr, epochs, batch_size, ticker
        - Returns: train_pred_df, val_pred_df, test_pred_df
    """
    # Datetime_col_name = 'Datetime' # May need to change this if the column name is Timestamp, not Datetime
    # raw_df.reset_index(drop=True, inplace=True)
    # n_window = n_past_days + n_future_days # 95 + 5 = 100
    # test set was made of 100 data points to predict 5 days using the past 95 days data

    # Split raw_df into train_df, val_df, and test_df
    train_df = raw_df.iloc[:-n_future_days-4,:] # use all available data for train and validation
    val_df = raw_df.iloc[-(n_past_days+n_future_days+4):,:] # validate on 5 samples
    test_df = raw_df.iloc[-n_past_days:,:]
    print(f'train_df.columns: \n {train_df.columns}')
    print(f'train_df.tail(): \n {train_df.tail()}')

    print(f'val_df.columns: \n {val_df.columns}')
    print(f'val_df.head(): \n {val_df.head()}')
    print(f'val_df.tail(): \n {val_df.tail()}')

    print(f'test_df.columns: \n {test_df.columns}')
    print(f'test_df.head(): \n {test_df.head()}')
    print(f'test_df.tail(): \n {test_df.tail()}')

    df_end_date = test_df.iloc[-1:,:]['Datetime'][0]
    print(f'df_end_date = {df_end_date}')


    #### Data Pre-processing ####

    ## Convert dataset (using multiple "selected" features (predictors)) to numpy array (matrix) format
    training_set_X = train_df[X_features].to_numpy()
    training_set_y = train_df[y_features].to_numpy()

    ## Feature scaling
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_X_scaled = X_scaler.fit_transform(training_set_X)
    train_y_scaled = y_scaler.fit_transform(training_set_y)

    ## Create a data structure with n_past_days past datetimes to predict n_future_days for all features
    X_train = []
    y_train = []
    datelist_train_pred = []
    datelist_train_pred_str = []

    for i in range(n_past_days, len(train_X_scaled) - n_future_days + 1):
        X_train.append(train_X_scaled[i-n_past_days:i, :])
        y_train.append(train_y_scaled[i:i+n_future_days, :])
        datelist_train_pred.append(list(raw_df['Datetime'][i:i+n_future_days]))
        datetime_str = [x.strftime('%Y-%m-%d') for x in raw_df['Datetime'][i:i+n_future_days]] 
        datelist_train_pred_str.append(datetime_str)
        
    print(f'datelist_train_pred[-1:] : \n {datelist_train_pred[-1:]}')
    # len(X_train) == len(training_set) - n_past_days - n_future_days + 1
    print(f'len(X_train) = {len(X_train)}') # eg., 1155 == (1254 - 95 - 5 + 1)

    ## Convert list to ndarray
    X_train, y_train = np.array(X_train), np.array(y_train)


    ## Apply the same data pre-processing procedures to validation data

    # convert dataset (using multiple "selected" features (predictors)) to numpy array (matrix) format
    val_set_X = val_df[X_features].to_numpy()
    val_set_y = val_df[y_features].to_numpy()

    # Scale the data
    val_X_scaled = X_scaler.transform(val_set_X)
    val_y_scaled = y_scaler.transform(val_set_y)

    # Create a data structure with n_past_days past datetimes to predict n_future_days for all features
    X_validation = []
    y_validation = []
    datelist_val_pred = []
    datelist_val_pred_str = []
    val_df_start_iloc = raw_df.index.get_loc(val_df.index[0])

    for i in range(n_past_days, len(val_X_scaled) - n_future_days + 1):
        X_validation.append(val_X_scaled[i-n_past_days:i, :])
        y_validation.append(val_y_scaled[i:i+n_future_days, :])
        datelist_val_pred.append(list(raw_df['Datetime'][val_df_start_iloc+i:val_df_start_iloc+i+n_future_days]))
        datetime_str = [x.strftime('%Y-%m-%d') for x in raw_df['Datetime'][val_df_start_iloc+i:val_df_start_iloc+i+n_future_days]] 
        datelist_val_pred_str.append(datetime_str)  
        
    print(f'datelist_val_pred[-1:] : \n {datelist_val_pred[-1:]}')
    # len(X_validation) == len(validation_set) - n_past_days - n_future_days + 1

    X_validation, y_validation = np.array(X_validation), np.array(y_validation)


    ## Apply the same data pre-processing procedures to test data

    # convert dataset (using multiple "selected" features (predictors)) to numpy array (matrix) format
    test_set_X = test_df[X_features].to_numpy()
    print(f'Shape of test_set_X == {test_set_X.shape}')

    test_X_scaled = X_scaler.transform(test_set_X)

    # reshape data to the format that is required for LSTM
    X_test = test_X_scaled.reshape(1, n_past_days, len(X_features))

    ## reshape y input from [samples, time steps, features] to [samples, features] which is required for LSTM
    y_train_reshaped = y_train.reshape(-1, n_future_days)
    y_val_reshaped = y_validation.reshape(-1, n_future_days)

    n_X_features = len(X_features) # 5
    n_y_features = len(y_features) # 1; 


    ####  Build LSTM ML model

    ## LSTM model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past_days, n_X_features)))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units= n_future_days*n_y_features)) # activation='linear' (i.e. None)

    ## Compile the model
    model.compile(optimizer = Adam(learning_rate=lr), loss=loss)
    print('%%%%%%%%%%%%%%%%%% model summary %%%%%%%%%%%%%%%%%%')
    model.summary()
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    ### Training the model

    ## EarlyStopping - Stop training when a monitored metric has stopped improving.
    # monitor - quantity to be monitored.
    # min_delta - minimum change in the monitored quantity to qualify as an improvement, 
    #             i.e. an absolute change of less than min_delta, will count as no improvement.
    # patience - number of epochs with no improvement after which training will be stopped.
    # ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.
    # factor - factor by which the learning rate will be reduced. new_lr = lr * factor.

    # es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown=5, verbose=1)
    # mcp = ModelCheckpoint(filepath=f'myLSTM_weights_{ticker}.h5', monitor='val_loss', verbose=1,
    #                         save_best_only=True, save_weights_only=True, mode='min')

    # model.fit(X_train, y_train_reshaped, validation_data=(X_validation,y_val_reshaped),
    #         epochs=epochs, callbacks=[es, rlr, mcp], batch_size=batch_size, verbose=1)

    model.fit(X_train, y_train_reshaped, validation_data=(X_validation,y_val_reshaped),
            epochs=epochs, callbacks=[rlr], batch_size=batch_size, verbose=1)


    #### Model Prediction for train, validation, and test data
    train_pred_scaled = model.predict(X_train)
    val_pred_scaled = model.predict(X_validation)
    test_pred_scaled = model.predict(X_test)

    #### Calculate RMSE performance metrics
    RMSE_train_pred_scaled = math.sqrt(mean_squared_error(y_train_reshaped, train_pred_scaled))
    RMSE_val_pred_scaled = math.sqrt(mean_squared_error(y_val_reshaped, val_pred_scaled))
    print(f'RMSE_train_pred_scaled = {RMSE_train_pred_scaled}')
    print(f'RMSE_val_pred_scaled = {RMSE_val_pred_scaled}')

    ### Check for RMSE in original data scale

    ## Inverse_transform the predicted values and scaled y_train input to the original format
    train_pred_scaleback =  y_scaler.inverse_transform(train_pred_scaled.reshape(-1,1)).reshape(-1,n_future_days)
    y_train_scaleback =  y_scaler.inverse_transform(y_train_reshaped.reshape(-1,1)).reshape(-1,n_future_days)

    RMSE_train_pred_scaleback = math.sqrt(mean_squared_error(y_train_scaleback, train_pred_scaleback))
    print(f'RMSE_train_pred_scaleback = {RMSE_train_pred_scaleback}')

    ## Inverse_transform the predicted values to the original format
    val_pred_scaleback =  y_scaler.inverse_transform(val_pred_scaled.reshape(-1,1)).reshape(-1,n_future_days)
    y_val_scaleback = y_scaler.inverse_transform(y_validation.reshape(-1,1)).reshape(-1,n_future_days)
    RMSE_val_pred_scaleback = math.sqrt(mean_squared_error(y_val_scaleback, val_pred_scaleback))
    print(f'RMSE_val_scaleback = {RMSE_val_pred_scaleback}')

    ## Inverse_transform the predicted values to the original format
    test_pred_scaleback = y_scaler.inverse_transform(test_pred_scaled.reshape(-1,1)).reshape(-1,n_future_days)
    print(f'test_pred_scaleback = {test_pred_scaleback}')


    ### Create a list of future days to predict: remove holidays

    us_cal = CustomBusinessDay(calendar=Calendar())

    df_end_date = test_df.iloc[-1:,:]['Datetime'][0]
    print(f'df_end_date = {df_end_date}')

    extra_days = n_future_days + int(n_future_days/3) + 10
    # print(f"extra_days = {extra_days}")

    extra_future_days = pd.date_range(start = (df_end_date + timedelta(days=1)),
                                    end = df_end_date + timedelta(days=extra_days),
                                    freq = us_cal)

    ## Create adjusted date list for test_pred_df
    datelist_test_pred = list(extra_future_days[:n_future_days])
    datelist_test_pred_str = list(extra_future_days[:n_future_days].strftime('%Y-%m-%d'))
    print(f'datelist_test_pred_str = {datelist_test_pred_str}')

    #### Finally, create train_pred_df, val_pre_df, and test_pred_df

    ## train_pred_df
    train_pred_df = pd.DataFrame(list(zip(train_pred_scaleback.tolist(), datelist_train_pred, datelist_train_pred_str)), 
                                    columns = ['Adj Close pred', 'Datetime_list', 'Datetime_str'])
    train_pred_df['RMSE'] = RMSE_train_pred_scaleback
    print(f'train_pred_df.head(): \n {train_pred_df.head(2)}')
    print(f'train_pred_df.tail(): \n {train_pred_df.tail(2)}')

    if len(train_pred_df) > 200:
        train_pred_df = train_pred_df.iloc[-200:,:] # limit train_pred_df length to 200 or less

    ## val_pre_df
    val_pred_df = pd.DataFrame(list(zip(val_pred_scaleback.tolist(), datelist_val_pred, datelist_val_pred_str)), 
                                columns = ['Adj Close pred', 'Datetime_list', 'Datetime_str'])
    val_pred_df['RMSE'] = RMSE_val_pred_scaleback
    print(f'val_pred_df.head(): \n {val_pred_df.head(2)}')
    print(f'val_pred_df.tail(): \n {val_pred_df.tail(2)}')

    ## test_pred_df
    test_pred_df = pd.DataFrame(list(zip(test_pred_scaleback.tolist(), [datelist_test_pred], [datelist_test_pred_str])), 
                                    columns = ['Adj Close pred', 'Datetime_list', 'Datetime_str'])
    print(f'test_pred_df: \n {test_pred_df}')

    return train_pred_df, val_pred_df, test_pred_df


if __name__ == '__main__':
    ## stock ticker list to investigate
    stock = ['GOOG']

    ### Part 1. Get raw stock data from mongodb and apply ML model to get future stock price prediction, 
    ###         then upload the model prediction results to mongodb

    ## Set start_date as 5 years from today: days = 1825 (365*5 = 1825 days)
    start_date = (datetime.now() - timedelta(days=1825)) # 262 business days/year * 5 years = 1310 business days 
    end_date = datetime.now()

    print(f"start_date = {start_date.strftime('%Y-%m-%d')}")
    print(f"end_date = {end_date.strftime('%Y-%m-%d')}")
    
    n_past_days = 90
    # n_future_days = 10

    # list(train_df.columns) # ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Datetime']
    # Select features (columns) to be included in training and predictions
    X_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    y_features = ['Adj Close']

    loss = 'mean_squared_error'
    lr = 0.01
    epochs = 20 
    batch_size = 10
    
    ## Download raw stock data from MongoDB and create raw_df
    for i in range(len(stock)):
        print(f'********** stock {stock[i]} raw data query *************')
        collection_name = f'{stock[i].lower()}_raw'
        print(f'stock {stock[i]} raw data collection_name = {collection_name}')
        # last_updated_date = db_last_update_date(ticker=stock[i], db_name='test_stock_raw', collection_name=collection_name)
        stock_query_list = mongodb_stock.stock_data_query(ticker=stock[i], db_name='test_stock_raw', 
                                                            collection_name=collection_name, past_days=5*365)
        raw_df = pd.DataFrame(stock_query_list).sort_values(by=['Datetime'], ascending=True)
        raw_df['Date'] = raw_df['Datetime']
        raw_df = raw_df.set_index('Date')
        # raw_df.reset_index(drop=True, inplace=True)

        stock_last_day = raw_df['Datetime'][-1]
        print(f'stock_last_day = {stock_last_day}')

        print(f'raw_df.columns: \n {raw_df.columns}')

        print(f'raw_df.info(): \n {raw_df.info()}')
        print(f'raw_df.head(): \n {raw_df.head()}')


        ### Stock price predictions for n_future_days using LSTM model

        n_future_days_list = [3]
        # n_future_days_list = [3, 10]
        
        for x in n_future_days_list:
            n_future_days = x
            print('+++++++++++++++++++++++++++++++++++++++++++++')
            print(f'n_future_days = {n_future_days}')
            # myLSTM(raw_df, X_features, y_features, n_past_days, n_future_days, loss='mean_squared_error',
            #  lr=0.01, epochs=12, batch_size=32)
            train_pred_df, val_pred_df, test_pred_df = myLSTM(raw_df, X_features, y_features, n_past_days, n_future_days,
                                                        loss=loss, lr=lr, epochs=epochs, batch_size=batch_size, ticker=stock[i])

            print(f'train_pred_df.columns: \n {train_pred_df.columns}')
            print(f'train_pred_df.head() = \n {train_pred_df.head()}')

            print(f'val_pred_df.columns: \n {val_pred_df.columns}')
            print(f'val_pred_df.head() = \n {val_pred_df.head()}')

            print(f'test_pred_df.columns: \n {test_pred_df.columns}')
            print(f'test_pred_df.head() = \n {test_pred_df.head()}')

            # use iloc since index doesn't start from zero if train_pred_df was sliced to the length of 200 or less
            RMSE_train_pred = train_pred_df['RMSE'].iloc[0] 
            # RMSE_train_pred = train_pred_df['RMSE'][0] # can use this if train_pred_df was NOT sliced earlier
            RMSE_val_pred = val_pred_df['RMSE'].iloc[0]
            print(f'RMSE_train_pred = {RMSE_train_pred}')
            print(f'RMSE_val_pred = {RMSE_val_pred}')

            val_pred_df_dict = val_pred_df.to_dict(orient='records')
            print(f'val_pred_df_dict: \n {val_pred_df_dict}')

            test_pred_df_dict = test_pred_df.to_dict(orient='records')
            print(f'test_pred_df_dict: \n {test_pred_df_dict}')

            try:            
                print(f'++ {stock[i]} stock price prediction data upload process begins ++++++++++++')

                ## Prepare model precition data for mongodb upload, i.e, transform them into dictionaries
          
                pred_post_to_upload = mongodb_stock.ml_pred_post(ticker=stock[i], stock_last_day=stock_last_day, 
                                        train_pred_df=train_pred_df, val_pred_df=val_pred_df, test_pred_df=test_pred_df, 
                                        X_features=X_features, y_features=y_features, n_past_days=n_past_days,
                                        n_future_days=n_future_days, loss=loss, lr=lr, epochs=epochs, batch_size=batch_size, 
                                        RMSE_train_pred=RMSE_train_pred, RMSE_val_pred=RMSE_val_pred)

                # print(f'pred_post_to_upload = {pred_post_to_upload}')

                ## Upload stock prediction data to mongodb
                mongodb_stock.stock_pred_upload(pred_post_to_upload, ticker = stock[i], db_name='test_stock_pred')
                # mongodb_stock.stock_pred_upload(pred_post_to_upload, ticker = stock[i], db_name='test_stock_pred')

                ## Send email notice
                email_subject = "Stock price prediction model Upload Notice"
                email_message = f'{stock[i]} Stock price prediction model for {n_future_days} future day(s) uploaded to MongoDB...\n\n No Error'
                sendEmailNotice(email_subject, email_message)
            except:
                ## Send email notice
                email_subject = "Stock price prediction model Upload Error Notice"
                email_message = f'{stock[i]} Stock price prediction model for {n_future_days} future day(s) NOT uploaded to MongoDB...'
                sendEmailNotice(email_subject, email_message)



    ### Part 2. retrieve stock price prediction data from mongodb
    # first, check the last date the ML pred data was uploaded, and query the lasted data
    for i in range(len(stock)):
        print(f'\n+++++++++++ {stock[i]} stock price prediction data query ++++++++++++\n')
        last_pred_updated_date = mongodb_stock.db_last_update_date(ticker=stock[i], db_name='stock_pred',
                                                                    collection_name=f'{stock[i].lower()}_pred')

        query_result = mongodb_stock.ml_pred_data_query(ticker=stock[i], db_name='stock_pred', 
                                                        collection_name=f'{stock[i].lower()}_pred',
                                                        stock_last_date=last_pred_updated_date)

        # find the model result with the smallest RMSE_val_pred value
        RMSE_val_pred_list = []

        for i in range(query_result.count()):            
            RMSE_val_pred_list.append(query_result[i]["RMSE_val_pred"])
            print(f'RMSE_train_pred {i} = {query_result[i]["RMSE_train_pred"]}')
            print(f'RMSE_val_pred {i} = {query_result[i]["RMSE_val_pred"]}')

        smallest_RMSE_index = RMSE_val_pred_list.index(min(RMSE_val_pred_list))
        print(f'smallest_RMSE_index = {smallest_RMSE_index}')

        # create test_pred_df using the query result with the smallest RMSE of the ml model
        test_pred = query_result[smallest_RMSE_index]["test_pred"][0]
        test_pred_datetime = test_pred["Datetime_list"]
        test_pred_datetime_str = test_pred["Datetime_str"]
        test_pred_adj_close = test_pred["Adj Close pred"]

        test_pred_df = pd.DataFrame(list(zip(test_pred_datetime, test_pred_adj_close, test_pred_datetime_str)),
                                        columns= ['Datetime', 'Adj Close pred', 'Datetime_str'])
        print(f'test_pred_df: \n {test_pred_df}')

        print(f'RMSE_train_pred = {query_result[smallest_RMSE_index]["RMSE_train_pred"]}')
        print(f'RMSE_val_pred = {query_result[smallest_RMSE_index]["RMSE_val_pred"]}')

