from asyncio import as_completed
from sklearn import preprocessing
import pandas as pd
import numpy as np
import talib as talib

BUY = -1
HOLD = 0
SELL = 1


class TecnicalAnalysis:

    @staticmethod
    def compute_oscillators(data):
        log_return = np.log(data['Close']) - np.log(data['Close'].shift(1))
        data['Z_score'] = (((log_return - log_return.rolling(20).mean()) / log_return.rolling(20).std()))
        data['RSI'] = ((talib.RSI(data['Close'])) / 100)
        upper_band, _, lower_band = talib.BBANDS(data['Close'], nbdevup=2, nbdevdn=2, matype=0)
        data['boll'] = ((data['Close'] - lower_band) / (upper_band - lower_band))
        data['ULTOSC'] = ((talib.ULTOSC(data['High'], data['Low'], data['Close'])) / 100)
        data['pct_change'] = (data['Close'].pct_change())
        data['zsVol'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
        data['PR_MA_Ratio_short'] = \
            ((data['Close'] - talib.SMA(data['Close'], 21)) / talib.SMA(data['Close'], 21))
        data['MA_Ratio_short'] = \
            ((talib.SMA(data['Close'], 21) - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))
        data['MA_Ratio'] = (
                    (talib.SMA(data['Close'], 50) - talib.SMA(data['Close'], 100)) / talib.SMA(data['Close'], 100))
        data['PR_MA_Ratio'] = ((data['Close'] - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))

        return data


    @staticmethod
    def add_timely_data(data):
        data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.dayofweek
        data['Month'] = pd.to_datetime(data['Date']).dt.month
        data['Hourly'] = pd.to_datetime(data['Date']).dt.hour / 4
        return data
    

    @staticmethod
    def assign_labels(data, b_window, f_window, alpha, beta):
        x = data.copy()
        x['Close_MA'] = x['Close'].ewm(span=b_window).mean()
        x['s-1'] = x['Close'].shift(-1 * f_window)
        x['alpha'] = alpha
        x['beta'] = beta * (1 + (f_window * 0.1))
        x['label'] = x.apply(TecnicalAnalysis.check_label, axis=1)
        return x['label']


    @staticmethod
    def check_label(z):
        if (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) > z['alpha']) and \
                (abs((z['s-1'] - z['Close_MA']) / z['Close_MA']) < (z['beta'])):
            if z['s-1'] > z['Close_MA']:
                return -1
            elif z['s-1'] < z['Close_MA']:
                return 1
            else:
                return 0
        else:
            return 0


    @staticmethod
    def find_patterns(x):
        x['CDL2CROWS'] = talib.CDL2CROWS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDL3INSIDE'] = talib.CDL3INSIDE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLBELTHOLD'] = talib.CDLBELTHOLD(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLDOJISTAR'] = talib.CDLDOJISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLENGULFING'] = talib.CDLENGULFING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHIKKAKE'] = talib.CDLHIKKAKE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLKICKING'] = talib.CDLKICKING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLLONGLINE'] = talib.CDLLONGLINE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLMARUBOZU'] = talib.CDLMARUBOZU(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLMATHOLD'] = talib.CDLMATHOLD(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLONNECK'] = talib.CDLONNECK(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLPIERCING'] = talib.CDLPIERCING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLSHORTLINE'] = talib.CDLSHORTLINE(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTHRUSTING'] = talib.CDLTHRUSTING(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLTRISTAR'] = talib.CDLTRISTAR(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(x['Open'], x['High'], x['Low'], x['Close']) / 100
        x['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        # x['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(x['Open'], x['High'], x['Low'], x['Close']) / 100
        return x
