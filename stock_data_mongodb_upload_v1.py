from credentials import mongodb_key
from email_notice_function import sendEmailNotice

from pymongo import MongoClient

from datetime import datetime, timedelta
import time
import pandas as pd

from pandas_datareader import data as pdr
import yfinance as yf

### Get ready to import stock data using pandas_datareader (via Yahoo!Finance)
yf.pdr_override()


def stock_data_mongodb_upload(stock=['GOOG'], upload_start_date=datetime(2015,1,1), 
upload_end_date=datetime.now(), db_name='test_stock_raw'):

    """
    # Step 1. Check the lastest upload date of the stock in mongodb, if any
    # Step 2: Query new stock data from yahoo!finance db for the selected time period
    # Step 3: Upload new stock data to mongodb (Cloud)
  
        - Args: stock, upload_start_date,upload_end_date, db_name
        - Returns: n/a
    """
    
    # Set db access
    db_client = MongoClient(f"mongodb+srv://{mongodb_key.username}:{mongodb_key.password}@clusterk.su3fg.azure.mongodb.net/<dbname>?ssl=true&ssl_cert_reqs=CERT_NONE&retryWrites=true&w=majority")
    stock_raw_db = db_client[db_name]
    stock_raw_db_collection_names = stock_raw_db.list_collection_names()
    print(f'Current stock_raw_db_collection_names = {stock_raw_db_collection_names}')

    # if input date is string, convert it to datetime type 
    if type(upload_start_date) == str:
        upload_start_date = datetime.strptime(upload_start_date, '%Y-%m-%d')
    else:
        pass
    if type(upload_end_date) == str:
        upload_end_date = datetime.strptime(upload_end_date, '%Y-%m-%d')
    else:
        pass

    ### Upload stock data for the selected time period
    for i in range(len(stock)):
        print(f'++++++++++++++++ Start of Query {i+1}: stock {stock[i]} ++++++++++++++++++++')
        ## set db collection (data table) to access
        stock_raw_collection_name = f'{stock[i].lower()}_raw' # goog_raw # amzn_raw
        stock_raw_collection = stock_raw_db[stock_raw_collection_name]

        if stock_raw_collection_name in stock_raw_db_collection_names:
            print(f"Getting ready to upload stock data to the existing MongoDB collection, {stock_raw_collection_name} ...")
            
            #### Step 1. check the lastest upload date of the stock in mongodb, if any 

            date_30_days_ago = upload_end_date-timedelta(days=30)
            # query stock data for the past 30 days from upload_end_date
            query_result = stock_raw_collection.find({ 'Stock':stock[i],
                                        'Datetime': {'$gte': date_30_days_ago, '$lte': upload_end_date}})

            if query_result.count() > 0:
                print(f'query_result.count() = {query_result.count()} for the past 30 days from {upload_end_date}')
                
            else:
                print('query_result.count() = 0 for the past 30 days')
                query_result = stock_raw_collection.find({ 'Stock':stock[i],
                                                'Datetime': {'$gte': datetime(2015,1,1), '$lte': upload_end_date}})

            result_date_list = []
            for x in list(query_result):
                result_date_list.append(x['Datetime'])            
            # print(f'result_date_list from the query = {result_date_list}')

            if len(result_date_list) == 0:
                print(f'result_date_list is empty!!!')
                collection_last_date = datetime(2015,1,1) #
                
            else:
                collection_last_date = max(result_date_list)
                print(f'mongodb collection_last_date = {collection_last_date}')
                
                
        else:
            print("Creating a new collection since it doesn't exist.......")
            print("Stock data between 2015-01-01 and today will be uploaded by default, unless selected otherwise.")
            collection_last_date = upload_start_date


        #### Step 2: Query new stock data from yahoo!finance db for the selected time period

        # Initialize raw_df as an empty dataframe which will be updated if relevant stock data exist
        raw_df = pd.DataFrame()

        if collection_last_date >= upload_end_date:
            print('Stock data for the selected time period already exist in MongoDB')
            print('Nothing to upload (No need to collect stock data from yahoo!finance)')

        else:
            yf_query_start_date = collection_last_date + timedelta(days=1)
            yf_query_end_date = upload_end_date + timedelta(days=1) # set today+1day 
            #! Note that yf_query_end_date is not included in yahoo stock price query,
            # just like python list index convention
            # But mongoDB query_end_date is included in the query            
            print(f'yf_query_start_date = {yf_query_start_date}')
            print(f'yf_query_end_date = {yf_query_end_date}')

            try:
                if yf_query_start_date.date() < yf_query_end_date.date():
                    # Get stock data from yahoo!finance for the selected time period
                    raw_df = pdr.get_data_yahoo(stock[i], yf_query_start_date, yf_query_end_date)
                    print(f'len(raw_df) = {len(raw_df)}')
                else:
                    print("Query start day should be at least one day earlier...")
                    print("No query from yahoo!finance will be made, thus, no data upload to MongoDB")
                    email_subject = "Stock Data Upload Error Notice"
                    email_message = f"No new query for {stock[i]} Stock data on {yf_query_start_date} is necessary...\n\n no data upload to MongoDB"
                    sendEmailNotice(email_subject, email_message)                             

            except:
                print(f'failed to download {stock[i]} stock data between {yf_query_start_date} and {yf_query_end_date}')
                

            #### Step 3: Upload new stock data to mongodb (Cloud)

            if len(raw_df) == 0:
                print('len(raw_df) == 0, Nothing to upload....')

            elif len(raw_df) > 0:
                try:
                    raw_df['Datetime'] = pd.to_datetime(raw_df.index)
                    raw_df['Stock'] = stock[i]
                    raw_df.drop_duplicates('Datetime', keep='first', inplace=True)
                    print(f'After removing duplicates, len(raw_df) = {len(raw_df)}')
                    print(f'raw_df.columns = {list(raw_df.columns)}')

                    # convert raw_df dataframe into a json-like object
                    db_upload_raw_data = raw_df.to_dict(orient='records')

                    # upload the stock data to MongoDB                        
                    stock_raw_collection.insert_many(db_upload_raw_data)
                    print(f'{stock[i]} stock data updated successfully to DB: {db_name}, Collection:{stock_raw_collection_name}')
                    time.sleep(5)
                    email_subject = "Stock Data Upload Notice"
                    email_message = f'{stock[i]} Stock data uploaded to MongoDB...\n\n No Error'
                    sendEmailNotice(email_subject, email_message) 

                except:
                    print("Error in raw_df.to_dict(orient='records')...\n No new db_update_raw_data..........")
                
            else:
                print('Strange..... Unknown error.....')
            
        print(f'================ End of Query {i+1}: stock {stock[i]} ========================\n')


if __name__ == '__main__':
    # Test run
    stock = ['GOOG']
    # stock = ['AAPL', 'GOOG']
    stock_data_mongodb_upload(stock, '2015-01-01', '2021-07-30', db_name='test_stock_raw')
    # stock_data_mongodb_upload(stock, datetime(2015,1,1), datetime.now(), db_name='test_stock_raw')

