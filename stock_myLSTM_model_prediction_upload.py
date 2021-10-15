### Import modules and packages
import pandas as pd
from datetime import datetime, timedelta 

# my modules
import mongodb_stock
from stock_ml_model import myLSTM
from email_notice_function import sendEmailNotice


def stock_myLSTM_model_prediction_upload(stock_list=['GOOG'], n_future_days_list=[1], n_past_days=90, 
loss='mean_squared_error', lr=0.01, epochs=25, batch_size=10, sendEmail=False):

    """
    # Step 1: Download raw stock data from mongodb
    # Step 2: Stock price predictions for n_future_days using myLSTM model
    # Step 3: Upload the model prediction results to mongodb
    # Step 4: Send email notice
       
        - Args: ticker, db_name, collection_name, past_days, loss, lr, epochs, batch_size, sendEmail
        - Returns: n/a
    """

    stock = stock_list

    #### Step 1: Download raw stock data from mongodb

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

        stock_last_day = raw_df['Datetime'][-1] #
        print(f'stock_last_day = {stock_last_day}')

        print(f'raw_df.columns: \n {raw_df.columns}')

        print(f'raw_df.info(): \n {raw_df.info()}')
        print(f'raw_df.head(): \n {raw_df.head()}')

   
        #### Step 2: Stock price predictions for n_future_days using myLSTM model

        ## Hyperparameters for LSTM ML model           
        # loss = 'mean_squared_error'
        # lr = 0.01
        # epochs = 25 
        # batch_size = 10

        n_future_days_list = n_future_days_list
        # n_future_days_list = [3, 10]
        
        for x in n_future_days_list:
            n_future_days = x
            print('+++++++++++++++++++++++++++++++++++++++++++++')
            print(f'n_future_days = {n_future_days}')

            train_pred_df, val_pred_df, test_pred_df = myLSTM(raw_df, X_features, y_features,
                                                            n_past_days, n_future_days,
                                                            loss=loss, lr=lr, epochs=epochs, 
                                                            batch_size=batch_size, ticker=stock[i])

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

            #### Step 3: Upload the model prediction results to mongodb

            try:            
                print(f'++ {stock[i]} stock price prediction data upload process begins ++++++++++++')
                ## Prepare model precition data for mongodb upload, i.e, transform them into dictionaries
                pred_post_to_upload = mongodb_stock.ml_pred_post(ticker=stock[i], stock_last_day=stock_last_day,
                train_pred_df=train_pred_df, val_pred_df=val_pred_df, test_pred_df=test_pred_df,
                X_features=X_features, y_features=y_features, n_past_days=n_past_days, n_future_days=n_future_days, 
                loss=loss, lr=lr, epochs=epochs, batch_size=batch_size, RMSE_train_pred=RMSE_train_pred,
                RMSE_val_pred=RMSE_val_pred)

                # print(f'pred_post_to_upload = {pred_post_to_upload}')

                ## Upload stock prediction data to mongodb
                mongodb_stock.stock_pred_upload(pred_post_to_upload, ticker = stock[i], db_name='test_stock_pred')
                # mongodb_stock.stock_pred_upload(pred_post_to_upload, ticker = stock[i], db_name='test_stock_pred')

                #### Step 4: Send email notice

                email_subject = "Stock price prediction model Upload Notice"
                email_message = f'{stock[i]} Stock price prediction model for {n_future_days} future day(s) uploaded to MongoDB...\n\n No Error'
                print(email_message)

                if sendEmail:
                    sendEmailNotice(email_subject, email_message)
                else:
                    print('No email notice was sent!')
            except:
                ## Send email notice
                email_subject = "Stock price prediction model Upload Error Notice"
                email_message = f'{stock[i]} Stock price prediction model for {n_future_days} future day(s) NOT uploaded to MongoDB...'
                print(email_message)

                if sendEmail:                
                    sendEmailNotice(email_subject, email_message)
                else:
                    print('No email notice')



if __name__ == "__main__":
    stock_myLSTM_model_prediction_upload(stock_list=['GOOG'], n_future_days_list=[3], 
    n_past_days=90, loss='mean_squared_error', lr=0.01, epochs=25, batch_size=16, sendEmail=True)

