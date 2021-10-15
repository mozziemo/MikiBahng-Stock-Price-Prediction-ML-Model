
from credentials import mongodb_key
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd

# Set db access
db_client = MongoClient(f"mongodb+srv://{mongodb_key.username}:{mongodb_key.password}@clusterk.su3fg.azure.mongodb.net/<dbname>?ssl=true&ssl_cert_reqs=CERT_NONE&retryWrites=true&w=majority")


def db_last_update_date(ticker='AMZN', db_name='test_stock_raw', collection_name='amzn_raw', query_end_date=datetime.now(), db_first_date=datetime(2015,1,1)):
    """
    # Check the last date of the existing stock data in DB   
        - Args: ticker, db_name, collection_name, query_end_date, db_first_date
        - Returns: collection_last_date (datetime object)
    """
    stock_db = db_client[db_name]
    stock_collection = stock_db[collection_name]

    if collection_name in stock_db.list_collection_names():

        date_30_days_ago = query_end_date-timedelta(days=30)
        # query stock data for the past 30 days from query_end_date
        query_result = stock_collection.find({'Stock':ticker,
                                            'Datetime': {'$gte': date_30_days_ago, '$lte': query_end_date}})

        if query_result.count() > 0:
            print(f'query_result.count() = {query_result.count()} for the past 30 days from {query_end_date}')
            
        else:
            print('query_result.count() = 0 for the past 30 days')
            query_result = stock_collection.find({ 'Stock':ticker,
                                            'Datetime': {'$gte': datetime(2015,1,1), '$lte': query_end_date}})

        result_date_list = []
        for x in list(query_result):
            result_date_list.append(x['Datetime'])            
        # print(f'result_date_list from the query = {result_date_list}')

        if len(result_date_list) == 0:
            print(f'result_date_list is empty!!!')
            
        else:
            collection_last_date = max(result_date_list)
            print(f'mongodb collection_last_date = {collection_last_date}')
            
            
    else:
        print("Creating a new collection since it doesn't exist.......")
        print("Stock data between 2015-01-01 and today will be uploaded by default, unless selected otherwise.")
        collection_last_date = db_first_date
 
    return collection_last_date



############


def stock_data_query(ticker=['AMZN'], db_name='test_stock_raw', collection_name='amzn_raw', past_days=5*365):

    """
    # Query raw stock data (for the past 5 years period, by default)   
        - Args: ticker, db_name, collection_name, past_days
        - Returns: raw_df
    """

    stock_db = db_client[db_name]
    stock_collection = stock_db[collection_name]

    if collection_name in stock_db.list_collection_names():
        past_days_start = datetime.now() - timedelta(days=past_days+5) #manually added 5 more days
        today = datetime.now()

        # query stock data for the past 30 days ago from today
        query_result = stock_collection.find({ 'Stock':ticker,
                                            'Datetime': {'$gte': past_days_start, '$lte': today}})
        
        raw_df = pd.DataFrame(list(query_result)).sort_values(by=['Datetime'], ascending=True)

    else:
        raw_df = pd.DataFrame({'Datetime':[]})
        print(f'Query process interrupted... No collection {collection_name} in DB {db_name} exists!!')

    return raw_df


############


def ml_pred_post(ticker=None, stock_last_day=None, train_pred_df=None, val_pred_df=None, test_pred_df=None, 
X_features=None, y_features=None, n_past_days=90, n_future_days=5, loss='mean_squared_error', lr=0.01, epochs=12, batch_size=32, RMSE_train_pred=None, RMSE_val_pred=None):

    """
    # Create a post (stock price prediction data with ML model parameters) for upload to MongoDB   
        - Args: ticker, stock_last_day, train_pred_df, val_pred_df, test_pred_df, 
                X_features, y_features, n_past_days, n_future_days, 
                loss, lr, epochs, batch_size, 
                RMSE_train_pred, RMSE_val_pred
        - Returns: pred_post_to_upload
    """

    try:
        train_pred_df_dict = train_pred_df.to_dict(orient='records')
        val_pred_df_dict = val_pred_df.to_dict(orient='records')
        test_pred_df_dict = test_pred_df.to_dict(orient='records')
        
        n_X_features = len(X_features)
        n_y_features = len(y_features)

        pred_post_to_upload = {
            'Stock': ticker,
            'Datetime': stock_last_day,
            'X_features': X_features,
            'y_features': y_features,
            'n_past_days': n_past_days,
            'n_future_days': n_future_days,
            'ML Model': {
                'model': 'LSTM',
                'parameters': {
                    'layers': {'LSTM_1 units': 64,
                            'LSTM_1 input_shape': (n_past_days, n_X_features),
                            'LSTM_2 units': 32,
                            'Dropout': 0.2,
                            'Dense units': n_future_days*n_y_features
                    },
                    'compile': {
                        'optimizer': 'Adam',
                        'loss': str(loss),
                        'lr': lr
                    },
                    'fit': {
                        'epochs': epochs,
                        'batch_size': batch_size
                    }   
                },

            },
            'RMSE_train_pred': float(RMSE_train_pred),
            'RMSE_val_pred': float(RMSE_val_pred),
            'train_pred': train_pred_df_dict,
            'val_pred': val_pred_df_dict,
            'test_pred': test_pred_df_dict
        }
        return pred_post_to_upload

    except Exception as e:
        print('Something went wrong...')
        print(f'str(e) = {str(e)}')
        # print(f'repr(e) = {repr(e)}')

    # return pred_post_to_upload


############


def stock_pred_upload(post, ticker = 'test_ticker', db_name='test_stock_pred'):

    """
    # Upload a post (stock price prediction data with ML model parameters) to MongoDB
        - Args: post, ticker, db_name
        - Returns: n/a
    """

    stock_pred_db = db_client[db_name]
    stock_pred_collection = stock_pred_db[f'{ticker.lower()}_pred']

    stock_pred_collection.insert_one(post)
    print(f'{ticker} stock pred data upload to MongoDB successfully!')
    
    stock_pred_db_collection_names = stock_pred_db.collection_names()
    print(f'stock_pred_db_collection_names = {stock_pred_db_collection_names}')



############


def ml_pred_data_query(ticker='AMZN', db_name='test_stock_pred', collection_name='amzn_pred', 
stock_last_date=None, n_future_days=5, n_past_days=90):

    """
    # Query ML Model Prediction Data for the selected conditions from MongoDB
        - Args: ticker, db_name, collection_name, stock_last_date, n_future_days, n_past_days
        - Returns: query_result
    """

    stock_db = db_client[db_name]
    stock_collection = stock_db[collection_name]
    
    if stock_last_date is str:
        stock_last_date = datetime.strptime(stock_last_date, '%Y-%m-%d')
    
    query_result =  stock_collection.find({'Stock': ticker,
                                          'Datetime': stock_last_date,
                                          'n_future_days': n_future_days,
                                          'n_past_days': n_past_days})
    return query_result


############


def ml_pred_data_query_count(ticker='AMZN', db_name='test_stock_pred', collection_name='amzn_pred', 
stock_last_date=None, n_future_days=5, n_past_days=90):

    """
    # Check how many ML Model Prediction Data for the selected conditions exist in MongoDB collection   
        - Args: ticker, db_name, collection_name, stock_last_date, n_future_days, n_past_days
        - Returns: query_result_count
    """

    stock_db = db_client[db_name]
    stock_collection = stock_db[collection_name]
    
    if stock_last_date is str:
        stock_last_date = datetime.strptime(stock_last_date, '%Y-%m-%d')
    
    query_result_count =  stock_collection.count_documents({'Stock': ticker,
                                          'Datetime': stock_last_date,
                                          'n_future_days': n_future_days,
                                          'n_past_days': n_past_days})

    return query_result_count


############


if __name__ == '__main__':

    print("+++++++++ Checking <db_last_update_date> function... +++++++++++++")

    ### Check the last date of the existing stock data in MongoDB collection

    collection_last_date = db_last_update_date(ticker='GOOG', db_name='test_stock_raw',
     collection_name='goog_raw', query_end_date=datetime.now())

    print(f'collection_last_date = {collection_last_date}')
    print(f'collection_last_date in string format = {collection_last_date.strftime("%Y-%m-%d")}')


    ### 

    print("\n++++ Checking <db_last_update_date> + <ml_pred_data_query> function... ++++\n")

    selected_stock = 'GOOG'
    n_future_days = 3

    last_pred_updated_date = db_last_update_date(ticker=selected_stock,
                           db_name='test_stock_pred', collection_name=f'{selected_stock.lower()}_pred')


    print(f'last_pred_updated_date = {last_pred_updated_date}')

    query_result = ml_pred_data_query(ticker=selected_stock, db_name='test_stock_pred', 
                                    collection_name=f'{selected_stock.lower()}_pred',
                                    stock_last_date=last_pred_updated_date, 
                                    n_future_days=n_future_days)


    # print(f'query_result = {list(query_result)}')
    print(f'query_result.count() = {query_result.count()}')

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
