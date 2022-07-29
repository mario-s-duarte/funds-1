import pandas as pd
from datetime import datetime

# Scrapping Exchange Rates from www.exchangerates.org.uk

url = 'http://www.exchangerates.org.uk/USD-EUR-exchange-rate-history.html'
csv = 'exchange_rates.csv'

df = pd.read_html(url, skiprows=[0,1,182])[0][[0,1]].rename(columns={0:'Date',1:'USD2EUR'})
df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x, infer_datetime_format = True).date())
df['USD2EUR'] = df['USD2EUR'].apply(lambda x: float(x[8:15]))
df.set_index('Date',inplace=True)

df_historic_usd2eur = pd.read_csv(csv,sep=';',parse_dates=True)
df_historic_usd2eur['Date'] = df_historic_usd2eur['Date'].apply(lambda x: pd.to_datetime(x, infer_datetime_format = True).date())
df_historic_usd2eur.set_index('Date',inplace=True)

df_historic_usd2eur = pd.concat([df.loc[:max(df_historic_usd2eur.index)][:-1],df_historic_usd2eur])
df_historic_usd2eur.to_csv(csv,sep=';')

# Update Portfolio From Transactions

transacoes_csv = 'transacoes.csv'

encoding='latin_1'
thousands = ','
decimal = '.'
to_date = lambda d: datetime.strptime(d, '%d-%m-%Y')
converters={'Data de subscrição': to_date}

portfolio_csv = 'portofolio.csv'
df_portofolio = pd.read_csv(transacoes_csv ,sep=';',encoding=encoding,thousands=thousands, decimal=decimal, converters=converters)
df_portofolio = df_portofolio[pd.isnull(df_portofolio['Data de resgate'])].groupby(['Code','Nome','Moeda'])['Quantidade'].sum().reset_index()
df_portofolio.to_csv(portfolio_csv,sep=';',index=False)

# Scrapping Quotes from FT

url = 'https://markets.ft.com/data/funds/tearsheet/historical?s={}:{}'
xls = 'historico_cotacoes.xlsx'

novo_dict_df = dict()

dict_df = {key.strip():value for key,value in pd.read_excel(xls, sheet_name=None).items()}

for symbol, _, currency, _ in df_portofolio.itertuples(index=False):
    try: # Get data from FT
        url = 'https://markets.ft.com/data/funds/tearsheet/historical?s={}:{}'.format(symbol,currency)
        df = pd.read_html(url)[0]
        df['Date'] = df['Date'].apply(lambda x: x[:-17]).apply(lambda x: pd.to_datetime(x, infer_datetime_format = True).date())
        df.set_index('Date',drop=True, inplace=True)
    except: # Get it from a csv dwoloaded from Morningstar
        path = '.\\cotacoes_morningstar\\{}.csv'.format(symbol)
        to_date = lambda d: datetime.strptime(d, '%Y-%m-%d')
        converters={'date': to_date}
        df = pd.read_csv(path ,sep=';',encoding=encoding,thousands=thousands, decimal=decimal, converters=converters)[['date','price']]
        df['date'] = df['date'].apply(lambda x: x.date())
        df = df.set_index('date',drop=True).sort_index(ascending=False).rename(columns={'price':'Close'})
        df.index.rename('Date',inplace=True)

    if symbol in dict_df.keys():
        dict_df[symbol]['Date'] = dict_df[symbol].apply(lambda x: pd.to_datetime(x['Date'], infer_datetime_format = True).date(), axis=1)
        dict_df[symbol].set_index('Date',drop=True, inplace=True)
        if df is not None:
            novo_dict_df[symbol] = pd.concat([df,dict_df[symbol][min(df.index):].iloc[1:]])
        else:
            print(symbol, 'Not Found in FT')
            novo_dict_df[symbol] = dict_df[symbol][:].iloc[1:]
    else:
        novo_dict_df[symbol] = df

writer = pd.ExcelWriter(xls)
for symbol, sheet in novo_dict_df.items():
    sheet.to_excel(writer, sheet_name=symbol)
writer.save()