import requests
import pandas as pd

class StockData:
    def __init__(self, csv_file_path: str, api_key: str) -> None:
        self.ticker_names = pd.read_csv(csv_file_path)['Symbol'].tolist()
        self.ticker_sectors = pd.read_csv(csv_file_path)['GICS Sector'].tolist()
        self.ticker_info = dict(zip(self.ticker_names, self.ticker_sectors))
        self.api_key = api_key
        self.base_url = 'https://eodhd.com/api/eod'

    def _fetch_data(self, ticker: str, period: str, start: str, end: str):
        url = f"{self.base_url}/{ticker}.US?period={period}&from={start}&to={end}&api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()  
        else:
            print(f"Failed to fetch data for {ticker}: {response.status_code}")
            return None
        

    def fetch_stock(self, ticker: str, period: str, start: str, end: str):
        print(f"fetch data for {ticker}...")
        result = self._fetch_data(ticker, period, start, end)
        df = pd.DataFrame(result)
        return df
    
    
    def fetch_all_stocks(self, sector: str, period: str, start: str, end: str):
        results = {}
        for ticker, s in self.ticker_info.items():
            if s == sector:
                results[ticker] = pd.DataFrame(self._fetch_data(ticker, period, start, end))

        all_data = self._process_merge_all_data(results)
        return all_data

    def _process_merge_all_data(self, dic:dict):
        all_data = []

        for name, df in dic.items():
            if df.empty:
                print(f"no data for {name}. skipping...")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df['ticker'] = name
            all_data.append(df)

        full_data = pd.concat(all_data)

        full_data.set_index(['date', 'ticker'], inplace=True)

        return full_data


if __name__ == "__main__":  
    stock = StockData('sp_400_midcap.csv', '662166cb8e3d13.57537943')
    # df = stock.fetch_stock(ticker = 'AA', period = 'd', start = '2010-01-01', end = '2010-07-01')
    df = stock.fetch_all_stocks(sector='Materials', period = 'd', start = '2010-01-01', end = '2010-07-01')
    print(df.head())
    print(df.shape)
        
        
