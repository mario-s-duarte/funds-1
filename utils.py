
import pandas as pd
from datetime import datetime

def load_data():
    
    #Get the Exchange Rates
    exchange_rates_csv = 'exchange_rates.csv'
    exchange_rates = pd.read_csv(exchange_rates_csv ,sep=';',parse_dates=[0])
    exchange_rates['Date'] = exchange_rates['Date'].apply(lambda x: x.date())
    exchange_rates = exchange_rates.set_index('Date')['USD2EUR']
    exchange_rates.head()

    df_quotes = pd.read_csv("ft_quotes.csv",index_col=0,parse_dates=True).sort_index(ascending=True).ffill().dropna(axis=1)
    df_quotes.index = df_quotes.index.date


    df_funds = pd.read_csv("ft_funds_perform.csv", index_col=0)[['name','currency']].rename(columns={'name':'Name','currency':'Currency'})
    df_funds = df_funds.loc[~df_funds.index.duplicated(keep='first')]


    ignore_list = {'LU1038809395','LU1799936510','LU0496443531','LU0439729285','LU0272942433','LU0442406459'}
    relevant_funds = set(df_funds[df_funds['Currency'].isin(["EUR","USD"])].index )
    relevant_funds &= set([idx for idx in df_funds.index if not idx.startswith("PT")])
    relevant_funds &= set(df_quotes.columns)
    relevant_funds -= set(ignore_list)

    df_etf_quotes = pd.read_csv("etf_quotes.csv", parse_dates=True,index_col=0).sort_index(ascending=True).ffill().bfill().dropna(axis=1)
    df_etf_quotes.index = df_etf_quotes.index.date

    df_etf_list = pd.read_csv('etf_list.csv',index_col=0)
    etf_ignore_list = ['FAGB.L','PHPD.AS']
    relevant_etfs = set(df_etf_list[df_etf_list['Currency'].isin(['EUR','USD'])].index )
    relevant_etfs &= set(df_etf_quotes.columns)
    relevant_etfs -= set(etf_ignore_list)

    df_quotes = pd.concat([df_quotes[list(relevant_funds)],df_etf_quotes[list(relevant_etfs)]],axis=1).dropna().sort_index(ascending=False)
    df_funds = pd.concat([df_funds,df_etf_list])

    for col in df_quotes.columns:
            if df_funds['Currency'][col] == 'USD':
                df_quotes[col] = df_quotes[col]* exchange_rates
    
    portofolio_cols = ['Code','Nome','Moeda']
    df_transacoes = pd.read_csv('transacoes.csv' ,parse_dates=True)
    df_portofolio = df_transacoes[pd.isnull(df_transacoes['Data de resgate'])].groupby(portofolio_cols)['Quantidade'].sum().reset_index()

    df_trans_etf = pd.read_csv('transacoes_etf.csv' ,parse_dates=True) 
    df_p_etf = df_trans_etf[pd.isnull(df_trans_etf['Data de resgate'])].groupby(portofolio_cols)['Quantidade'].sum().reset_index()

    df_portofolio = pd.concat([df_portofolio,df_p_etf]).set_index('Code').drop('LU0122613903')
    

    # Criar Dataframe com o Historico
    df_historico = pd.concat([
        df_transacoes[pd.notnull(df_transacoes['Data de resgate'])],
        df_trans_etf[pd.notnull(df_trans_etf['Data de resgate'])]])

    df_historico['Data de subscricao'] = df_historico['Data de subscricao'].apply(lambda d: datetime.strptime(d, '%d-%m-%Y').date())
    df_historico['Data de resgate'] = df_historico['Data de resgate'].apply(lambda d: datetime.strptime(d, '%d-%m-%Y').date())

    df_historico['Dias'] = (df_historico['Data de resgate']-df_historico['Data de subscricao']).apply(lambda x: x.days)
    df_historico['Cotacao de Subscricao'] = df_historico['Cotacao de Subscricao'].astype(float)
    df_historico['Cotacao de resgate'] = df_historico['Cotacao de resgate'].astype(float)

    return df_portofolio, df_quotes, exchange_rates, df_historico

