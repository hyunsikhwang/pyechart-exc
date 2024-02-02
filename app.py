from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import streamlit as st
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
import requests
import bs4
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from pymongo import MongoClient
from pykrx import stock
from datetime import datetime, timedelta
from pytz import timezone, utc
import plotly.express as px
import plotly.io as pio
import yfinance as yf
import os
os.system('pip install -U kaleido')

st.set_page_config(page_title="CNN Fear and Greed Index", layout="wide", page_icon="random")

st.header("pyechart")


url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

def get_bs(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.get(url, headers=headers).text, "lxml")

def get_bs_KRX(url, params):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    headers = {
                'Host': 'data.krx.co.kr',
                'Connection': 'keep-alive',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'ko,en-US;q=0.9,en;q=0.8',
    }
    return bs4.BeautifulSoup(requests.get(url, headers=headers, params=params).text, "lxml")


tab1, tab2, tab3, tab4 = st.tabs(["Fear and Greed Index", "Buffet Index(Korea)", "Dashboard", "Treemap(Korea)"])

with tab1:
    response = get_bs(url)

    x_axis = []
    y_axis = []

    line = Line()

    for itm in json.loads(response.text)['fear_and_greed_historical']['data']:
        x_axis.append(datetime.fromtimestamp(itm['x']/1000).strftime('%Y-%m-%d'))
        y_axis.append(itm['y'])

    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="800px"))
        .add_xaxis(x_axis)
        .add_yaxis('Index',
                y_axis,
                markpoint_opts=opts.MarkPointOpts(
                    data=[opts.MarkPointItem(name="Current", type_=None, coord=[x_axis[-1], y_axis[-1]], value=f"{y_axis[-1]:.1f}")]),
                is_smooth=True,
                is_step=False,
                label_opts=opts.LabelOpts(
                        formatter=JsCode(
                            "function(params){return params.value[1].toFixed(1);}"
                        )
                ),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
        )
        .set_series_opts(markarea_opts=opts.MarkAreaOpts(
                data=[
                    opts.MarkAreaItem(name="EXTREME FEAR", y=(0, 25), itemstyle_opts=opts.ItemStyleOpts(color="red", opacity=0.2)),
                    opts.MarkAreaItem(name="FEAR", y=(25, 45), itemstyle_opts=opts.ItemStyleOpts(color="orange", opacity=0.2)),
                    opts.MarkAreaItem(name="NEUTRAL", y=(45, 55), itemstyle_opts=opts.ItemStyleOpts(color="yellow", opacity=0.2)),
                    opts.MarkAreaItem(name="GREED", y=(55, 75), itemstyle_opts=opts.ItemStyleOpts(color="green", opacity=0.2)),
                    opts.MarkAreaItem(name="EXTREME GREED", y=(75, 100), itemstyle_opts=opts.ItemStyleOpts(color="blue", opacity=0.2)),
                ],
        ))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="CNN Fear and Greed Index"),
            tooltip_opts=opts.TooltipOpts(
                formatter=JsCode(
                    "function (params) {return params.value[0] + '<br>' + params.value[1].toFixed(1);}"
                )
            ),
            xaxis_opts=opts.AxisOpts(interval=0,
                                    boundary_gap=False,)
        )
        .render_embed()
    )
    #line.render()

    #st_pyecharts(line)
    components.html(line, height=800)


# Buffet Index


def post_bs(url, payload):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.post(url, headers=headers, data=payload).text, "lxml")

# 주가지수 조회
def idx_prc(mktType):
 
    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
 
    end_dd = datetime.today().strftime("%Y%m%d")
    strt_dd = (datetime.now() - relativedelta(years=10)).strftime("%Y%m%d")

    if mktType == 'KOSPI':
        param = ['코스피', '1']
    elif mktType == 'KOSDAQ':
        param = ['코스닥', '2']
    else:
        param = ['코스피', '1']
 
    payload = {
               'bld': 'dbms/MDC/STAT/standard/MDCSTAT00301',
               'tboxindIdx_finder_equidx0_7': param[0],
               'indIdx': param[1],
               'indIdx2': '001',
               'codeNmindIdx_finder_equidx0_7': param[0],
               'param1indIdx_finder_equidx0_7':'',
               'strtDd': strt_dd,
               'endDd': end_dd,
               'share': '2',
               'money': '3',
               'csvxls_isNo':'false'
    }
    MktData = post_bs(url, payload)

    try:
        data = json.loads(MktData.text)
    except:
        st.error(MktData.text)
 
    elevations = json.dumps(data['output'])
    day_one = pd.read_json(elevations)
    org_df = pd.DataFrame(day_one)
 
    return org_df


ecos_api_key = st.secrets["ecos_api_key"]
mgdb_id = st.secrets["mgdb_id"]
mgdb_pw = st.secrets["mgdb_pw"]

ecos_url = 'http://ecos.bok.or.kr/api'

now = datetime.now()
currYQ = f'{now.year}Q{(now.month-1)//3+1}'

url = f'{ecos_url}/StatisticSearch/{ecos_api_key}/json/kr/1/10000/200Y005/Q/2000Q1/{currYQ}/1400'

ecos_result = get_bs(url)
ecos_json = json.loads(ecos_result.text)

df = pd.json_normalize(ecos_json['StatisticSearch']['row'])
df = df[['TIME', 'DATA_VALUE']]
df['TIME'] = df['TIME'].str.replace('Q', '')
#display(df)
df['TIME'] = pd.to_datetime(df['TIME'].str[:4] + (df['TIME'].str[4:].astype(int)*3).astype(str).str.zfill(2) + '01')
df['TIME'] = df['TIME'].apply(lambda x: MonthEnd().rollforward(x))
df['DATA_VALUE'] = df['DATA_VALUE'].astype('float64')
df['AnnSum'] = df['DATA_VALUE'].rolling(4).sum() * 1000000000

# interpolation
df = df.set_index('TIME').resample('D').interpolate(method='cubic').reset_index()
# datetime 형태 변경
df['TIME'] = df['TIME'].dt.strftime('%Y-%m-%d')


# df_idx_KOSPI = idx_prc('KOSPI')
MONGO_DB = "cluster0"
dataset = 'MV_KOSPI'
connection = MongoClient(f'mongodb://{mgdb_id}:{mgdb_pw}@cluster0-shard-00-00-k5utu.mongodb.net:27017,cluster0-shard-00-01-k5utu.mongodb.net:27017,cluster0-shard-00-02-k5utu.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin')
collection = connection[MONGO_DB][dataset]
df_idx_KOSPI = pd.DataFrame(list(collection.find()))[['TRD_DD', 'MKTCAP_KOSPI', 'MKTCAP', 'CLSPRC_IDX']]

df_idx_KOSPI['MKTCAP'] = df_idx_KOSPI['MKTCAP'].replace({',':''}, regex=True).astype(float)
# df_idx_KOSDAQ = idx_prc('KOSDAQ')
# df_idx_KOSDAQ['MKTCAP'] = df_idx_KOSDAQ['MKTCAP'].replace({',':''}, regex=True).astype(float)

# df_idx_prc = df_idx_KOSPI.merge(df_idx_KOSDAQ, how='left', left_on='TRD_DD', right_on='TRD_DD')
# df_idx_prc['MKTCAP'] = df_idx_prc['MKTCAP_x'] + df_idx_prc['MKTCAP_y']

df_idx_prc = df_idx_KOSPI.copy()
df_idx_prc['TRD_DD'] = pd.to_datetime(df_idx_prc['TRD_DD']).dt.strftime('%Y-%m-%d')
# df_idx = df_idx_prc[['TRD_DD', 'MKTCAP_x', 'MKTCAP', 'CLSPRC_IDX_x']].copy()
# df_idx.rename({'MKTCAP_x':'MKTCAP_KOSPI', 'CLSPRC_IDX_x':'CLSPRC_IDX'}, axis='columns', inplace=True)

df_new = pd.merge(df_idx_prc, df, how='left', left_on='TRD_DD', right_on='TIME')
#display(df_new.dtypes)
df_new['Ratio'] = df_new['MKTCAP'] / df_new['AnnSum']
df_new['Ratio_KOSPI'] = df_new['MKTCAP_KOSPI'] / df_new['AnnSum']



df_new['TRD_DD'] = pd.to_datetime(df_new['TRD_DD'], format="%Y-%m-%d")

#df_new = df_new.set_index('TIME').resample('D').interpolate(method='cubic').reset_index()
# st.write(df_new)
df_new = df_new.drop_duplicates(['TRD_DD'])
df_new = df_new.set_index('TRD_DD').resample('D').interpolate(method='cubic').reset_index()
df_new['DATA_VALUE'] = df_new['DATA_VALUE'].interpolate(method='linear')
df_new['AnnSum'] = df_new['AnnSum'].interpolate(method='linear')
df_new['Ratio'] = df_new['MKTCAP'] / df_new['AnnSum']
df_new['Ratio_KOSPI'] = df_new['MKTCAP_KOSPI'] / df_new['AnnSum']

df_new['TRD_DD'] = df_new['TRD_DD'].dt.strftime('%Y-%m-%d')

#df_new.to_excel('gdp.xlsx')



# 1. 최초시점 비율을 기준으로 함
# 2. 매년 1월 1일자 위치에 "최초시점 비율" * GDP(AnnSum) 입력
# 2-1. 맨 처음 record 와 맨 마지막 record 에도 위의 수식 적용
# 3. 연간 값은 linear interpolation 처리


# 최초시점 비율
#initRatio = df_new.loc[0, 'Ratio_KOSPI']

with tab2:
    initRatio = df_new['Ratio_KOSPI'].median()
    maxRatio = df_new['Ratio_KOSPI'].max()

    Ratio_max = min(df_new['Ratio_KOSPI'].max()*0.9, initRatio * 3)
    Ratio_min = max(df_new['Ratio_KOSPI'].min()*1.2, initRatio * 0.6)
    Ratio_high = (initRatio + Ratio_max) * 0.5
    Ratio_low = (initRatio + Ratio_min) * 0.5
    band = f'{initRatio:.0%} base'

    # Buffet Index Range
    Ratio_max = 1.14
    Ratio_min = 0.61
    Ratio_high = 0.97
    Ratio_low = 0.79

    selPeriod = st.selectbox('Select Year', ['All', '1', '2', '3', '4', '5', '10'], index=5)

    df_new['TRD_DD'] = pd.to_datetime(df_new['TRD_DD'])
    df_new[band] = np.where((df_new['TRD_DD'].dt.month == 1) & (df_new['TRD_DD'].dt.day == 1), initRatio * df_new['AnnSum'], np.nan)
    df_new.loc[0, band] = initRatio * df_new.loc[0, 'AnnSum']
    df_new[band] = df_new[band].interpolate(method='linear')

    bandMin = f'{Ratio_min:.0%} Modestly Undervalued'
    bandLow = f'{Ratio_low:.0%} Fair Valued'
    bandHigh = f'{Ratio_high:.0%} Modestly Overvalued'
    bandMax = f'{Ratio_max:.0%} Significantly Overvalued'

    df_new[bandMin] = df_new[band] * Ratio_min / initRatio
    df_new[bandLow] = df_new[band] * Ratio_low / initRatio
    df_new[bandHigh] = df_new[band] * Ratio_high / initRatio
    df_new[bandMax] = df_new[band] * Ratio_max / initRatio

    df_new = df_new.ffill()

    if selPeriod != 'All':
        end_dd = datetime.today().strftime("%Y-%m-%d")
        strt_dd = (datetime.now() - relativedelta(years=int(selPeriod))).strftime("%Y-%m-%d")
        df_new = df_new[(df_new['TRD_DD'].between(strt_dd, end_dd))]

    x_data = list(df_new['TRD_DD'].dt.strftime("%Y-%m-%d"))
    y_data = list(df_new['MKTCAP_KOSPI'])
    y_data_ratio = list(df_new['Ratio_KOSPI'])
    y_max = df_new['MKTCAP_KOSPI'].max()*1.1

    y_bandMin = list(df_new[bandMin])
    y_bandLow = list(df_new[bandLow]-df_new[bandMin])
    y_bandHigh = list(df_new[bandHigh]-df_new[bandLow])
    y_bandMax = list(df_new[bandMax]-df_new[bandHigh])
    y_bandLimit = list(y_max-df_new[bandMax])

    if y_data_ratio[-1] >= Ratio_max:
        y_status = 'Significantly Overvalued'
    elif y_data_ratio[-1] >= Ratio_high:
        y_status = 'Modestly Overvalued'
    elif y_data_ratio[-1] >= Ratio_low:
        y_status = 'Fair Valued'
    elif y_data_ratio[-1] >= Ratio_min:
        y_status = 'Modestly Undervalued'
    else:
        y_status = 'Significantly Undervalued'


    st.subheader("Buffet Index for Korea Stock Market (KOSPI)")

    line_buffet = (
        Line(init_opts=opts.InitOpts(width="100%", height="800px"))
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
            series_name="",
            y_axis=y_data,
            symbol="emptyCircle",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True,
            markpoint_opts=opts.MarkPointOpts(
                data=[opts.MarkPointItem(name="Current", type_=None, coord=[x_data[-1], y_data[-1]], value=f"{y_data_ratio[-1]:,.0%}\n\n{y_status}")]),
        )
        .add_yaxis(
            series_name="Significantly Undervalued",
            y_axis=y_bandMin,
            symbol="emptyCircle",
            stack="Total",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="blue"),
            linestyle_opts=opts.LineStyleOpts(opacity=0),
        )
        .add_yaxis(
            series_name="Modestly Undervalued",
            y_axis=y_bandLow,
            symbol="emptyCircle",
            stack="Total",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="green"),
            linestyle_opts=opts.LineStyleOpts(opacity=0),
        )
        .add_yaxis(
            series_name="Fair Valued",
            y_axis=y_bandHigh,
            symbol="emptyCircle",
            stack="Total",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="yellow"),
            linestyle_opts=opts.LineStyleOpts(opacity=0),
        )
        .add_yaxis(
            series_name="Modestly Overvalued",
            y_axis=y_bandMax,
            symbol="emptyCircle",
            stack="Total",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="orange"),
            linestyle_opts=opts.LineStyleOpts(opacity=0),
        )
        .add_yaxis(
            series_name="Significantly Overvalued",
            y_axis=y_bandLimit,
            symbol="emptyCircle",
            stack="Total",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="red"),
            linestyle_opts=opts.LineStyleOpts(opacity=0),
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                max_=y_max,
                axistick_opts=opts.AxisTickOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                boundary_gap=False,
                interval=0),
        )

        .render_embed()
    )

    components.html(line_buffet, height=800)

class vlStatus:

    def __init__(self, market):
        KST = timezone('Asia/Seoul')
        now = datetime.utcnow()

        self.dtNow = utc.localize(now).astimezone(KST).strftime('%Y%m%d')
        self.dtPrev = f"{int(utc.localize(now).astimezone(KST).strftime('%Y'))-1}-12-01"
        self.dtThisYear = int(f"{int(utc.localize(now).astimezone(KST).strftime('%Y'))}")

    def kr(self, quote, quoteName):

        self.dtPrev = self.dtPrev.replace('-', '')
        df_idx = stock.get_index_fundamental(self.dtPrev, self.dtNow, quote).reset_index()

        df_idx_max = df_idx.groupby(pd.DatetimeIndex(df_idx['날짜']).year, as_index=False).agg({'날짜': max}).reset_index(drop=True)
        df_idx2 = pd.merge(df_idx, df_idx_max, how='right', on='날짜')[['날짜', '종가']]
        df_idx2['pct_change'] = df_idx2['종가'].pct_change(periods=1, axis='rows')

        df_idx.rename(columns={'날짜':'Date', '종가':'Close'}, inplace=True)

        date = pd.to_datetime(df_idx2.tail(1)['날짜'].values[0]).strftime("%Y년 %m월 %d일")
        value_closed = df_idx2.tail(1)['종가'].values[0]
        change_closed = df_idx2.tail(1)['pct_change'].values[0]

        updn = '상승' if change_closed > 0 else '하락'
        res_str = f"{quote} 의 {date} 기준 종가는 {value_closed:,.2f} 이며, 연초대비 {change_closed:.2%} {updn}하였습니다."

        df_idx = df_idx[['Date', 'Close']]
        df_idx.loc[:, 'quote'] = quote
        df_idx.loc[:, 'quoteName'] = quoteName
        df_idx = df_idx[(df_idx['Date']>=df_idx.groupby(pd.DatetimeIndex(df_idx['Date']).year, as_index=False).agg({'Date': max}).reset_index(drop=True)['Date'].head(1).values[0])]
        df_idx['changepct'] = df_idx['Close'] / df_idx.head(1)['Close'].values[0]

        df_idx = df_idx.reset_index(drop=True)
        df_idx.loc[0, 'Date'] = f'{self.dtThisYear}-01-01'

        return res_str, df_idx

    def us(self, quote, quoteName):
        idx = yf.Ticker(quote)
        df_idx = idx.history(start=self.dtPrev).reset_index()

        df_idx_max = df_idx.groupby(pd.DatetimeIndex(df_idx['Date']).year, as_index=False).agg({'Date': max}).reset_index(drop=True)
        df_idx2 = pd.merge(df_idx, df_idx_max, how='right', on='Date')[['Date', 'Close']]
        df_idx2['pct_change'] = df_idx2['Close'].pct_change(periods=1, axis='rows')

        date = pd.to_datetime(df_idx2.tail(1)['Date'].values[0]).strftime("%Y년 %m월 %d일")
        value_closed = df_idx2.tail(1)['Close'].values[0]
        change_closed = df_idx2.tail(1)['pct_change'].values[0]

        updn = '상승' if change_closed > 0 else '하락'
        res_str = f"{quote} 의 {date} 기준 종가는 {value_closed:,.2f} 이며, 연초대비 {change_closed:.2%} {updn}하였습니다."

        df_idx['Date'] = df_idx['Date'].dt.tz_localize(None)
        df_idx = df_idx[['Date', 'Close']]
        df_idx.loc[:, 'quote'] = quote
        df_idx.loc[:, 'quoteName'] = quoteName
        df_idx = df_idx[(df_idx['Date']>=df_idx.groupby(pd.DatetimeIndex(df_idx['Date']).year, as_index=False).agg({'Date': max}).reset_index(drop=True)['Date'].head(1).values[0])]
        df_idx['changepct'] = df_idx['Close'] / df_idx.head(1)['Close'].values[0]

        df_idx = df_idx.reset_index(drop=True)
        df_idx.loc[0, 'Date'] = f'{self.dtThisYear}-01-01'

        return res_str, df_idx



with tab3:
    quotes = ['1001', '1028', '2001', '2203']

    st.subheader("Dashboard")

    KST = timezone('Asia/Seoul')
    now = datetime.utcnow()

    SeoulTime = utc.localize(now).astimezone(KST).strftime('%Y%m%d')
    SeoulTime_b7d = (utc.localize(now).astimezone(KST) - timedelta(days=7)).strftime('%Y%m%d')

    df_KS = stock.get_index_price_change(f"{SeoulTime[:4]}0101", SeoulTime, "KOSPI").reset_index()
    df_KQ = stock.get_index_price_change(f"{SeoulTime[:4]}0101", SeoulTime, "KOSDAQ").reset_index()

    df_KR = pd.concat([df_KS[(df_KS['지수명'].isin(['코스피', '코스피 200']))], df_KQ[(df_KQ['지수명'].isin(['코스닥', '코스닥 150']))]]).reset_index(drop=True)[['지수명', '시가', '종가', '등락률']]

    df_vals = pd.DataFrame()
    for quote in quotes:
        df_val = stock.get_index_fundamental(f"{SeoulTime[:4]}0101", SeoulTime, quote).reset_index()
        df_val = df_val.tail(1)[['PER', 'PBR']]
        df_vals = pd.concat([df_vals, df_val])

    df_vals = df_vals.reset_index(drop=True)
    df_KR = pd.concat([df_KR, df_vals], axis=1)

    st.write(df_KR)
    # st.write(df_vals)

    quotes_index = {'kr':{"KOSPI":"1001",
                          "KOSDAQ":"2001"},
                    'us': {"S&P 500":'^GSPC',
                           "Dow Jones":'^DJI',
                           "NASDAQ":'^IXIC',
                           "Russell 2000":'^RUT',
                           "Nikkei 225":'^N225'}
                    }

    df = pd.DataFrame()

    for mkt in quotes_index.keys():
        x = vlStatus(market=mkt)
        for quote, quoteName in zip(quotes_index[mkt].values(), quotes_index[mkt].keys()):
            # print(f"x.{mkt}('{quote}')[1]")
            df_tmp = eval(f"x.{mkt}('{quote}', '{quoteName}')[1]")
            df = pd.concat([df, df_tmp])
    
    fig = px.line(df,
              x='Date',
              y='changepct',
              color='quoteName',
              line_shape='spline',
              markers=False,
            #   animation_frame='ix'
              )
    fig.update_xaxes(dtick="D1",
                    zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    fig.update_layout(width=1200)
    fig.add_shape( # add a horizontal "target" line
        type="line", line_color="salmon", line_width=2, opacity=1, line_dash="dot",
        x0=0, x1=1, xref="paper", y0=1, y1=1, yref="y"
    )

    st.plotly_chart(fig, use_container_width=False)



def maxworkdt_command():
 
    url = 'http://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd'
    params = {'baseName': 'krx.mdc.i18n.component',
              'key': 'B128.bld',
              'menuId': 'MDC0201030108'}

    MktData = get_bs_KRX(url, params=params)

    data = json.loads(MktData.text)
    
    df_result = data['result']['output'][0]['max_work_dt']
 
    return df_result

# 전종목 등락률
def KRX_12002(schdate, period):
    df_result = pd.DataFrame()
 
    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'

    # 1일
    # 1주
    # 1개월
    # 1분기
    # 6개월
    # 1년
    # 3년
    # 5년

    f = '%Y%m%d'

    period = period.upper()
    print(period)

    if period == 'D':
        strtDd = (datetime.strptime(schdate, f)).strftime(f)
    elif period == 'W':
        strtDd = (datetime.strptime(schdate, f) - timedelta(weeks=1) + timedelta(days=1)).strftime(f)
    elif period == 'M':
        strtDd = (datetime.strptime(schdate, f) - relativedelta(months=1) + timedelta(days=1)).strftime(f)
    elif period == 'Q':
        strtDd = (datetime.strptime(schdate, f) - relativedelta(months=3) + timedelta(days=1)).strftime(f)
    elif period == 'HY':    
        strtDd = (datetime.strptime(schdate, f) - relativedelta(months=6) + timedelta(days=1)).strftime(f)
    elif period == '1Y' or period == 'Y':
        strtDd = (datetime.strptime(schdate, f) - relativedelta(years=1) + timedelta(days=1)).strftime(f)
    elif period == '3Y':
        strtDd = (datetime.strptime(schdate, f) - relativedelta(years=3) + timedelta(days=1)).strftime(f)
    else:
        strtDd = (datetime.strptime(schdate, f)).strftime(f)
    print(strtDd, schdate)
 
    payload = {'bld': 'dbms/MDC/STAT/standard/MDCSTAT01602',
               'mktId': 'ALL',
               'strtDd': strtDd,
               'endDd': schdate,
               'adjStkPrc_check': 'Y',
               'adjStkPrc': '2',
               'share': '1',
               'money': '1',
               'csvxls_isNo': 'false'
    }
    MktData = post_bs(url, payload)

    data = json.loads(MktData.text)
    #display(pd.DataFrame(data['block1']))

    df_result = pd.DataFrame(data['OutBlock_1'])

    while df_result.BAS_PRC.str.replace(',', '').replace('-', '0').astype(float).sum() == 0:
        strtDd = (datetime.strptime(strtDd, f) - timedelta(days=1)).strftime(f)
        print(strtDd, schdate)
        payload = {'bld': 'dbms/MDC/STAT/standard/MDCSTAT01602',
                'mktId': 'ALL',
                'strtDd': strtDd,
                'endDd': schdate,
                'adjStkPrc_check': 'Y',
                'adjStkPrc': '2',
                'share': '1',
                'money': '1',
                'csvxls_isNo': 'false'
        }
        MktData = post_bs(url, payload)


        data = json.loads(MktData.text)
        #display(pd.DataFrame(data['block1']))

        df_result = pd.DataFrame(data['OutBlock_1'])
 
    return df_result, strtDd


# 업종분류현황
def KRX_12025(schdate):
    df_result = pd.DataFrame()
 
    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'

    mktIds = ['STK', 'KSQ']

    df_result = pd.DataFrame()

    for mktId in mktIds:
 
        payload = {'bld': 'dbms/MDC/STAT/standard/MDCSTAT03901',
                    'mktId': mktId,
                    'trdDd': schdate,
                    'money': '1',
                    'csvxls_isNo': 'false'
        }
        MktData = post_bs(url, payload)

        data = json.loads(MktData.text)
        #display(pd.DataFrame(data['block1']))

        df = pd.DataFrame(data['block1'])

        df_result = pd.concat([df_result, df])
 
    return df_result    

# 전종목 Valuation 조회
def all_val(end_dd):

    url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
 
    strt_dd = (datetime.now() - relativedelta(days=10)).strftime("%Y%m%d")

    payload = {'bld':'dbms/MDC/STAT/standard/MDCSTAT03501',
               'searchType':'1',
               'mktId': 'ALL',
               'trdDd': end_dd,
               'tboxisuCd_finder_stkisu0_0': '005930%2F%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90',
               'isuCd': 'KR7005930003',
               'isuCd2': 'KR7005930003',
               'codeNmisuCd_finder_stkisu0_0': '%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90',
               'param1isuCd_finder_stkisu0_0': 'STK',
               'strtDd': strt_dd,
               'endDd': end_dd,
               'csvxls_isNo': 'false',
              }
 
    MktData = post_bs(url, payload)
 
    data = json.loads(MktData.text)
    #display(data['output'])
    df_result = pd.DataFrame(data['output'])
 
    return df_result

def TreeMap(mktType):
    dict_Metric = {'Price':['FLUC_RT', 0],
                'PER':['PER', 1],
                'PBR':['PBR', 2],
                'Dividend':['DVD_YLD', 3],
                'ROE':['ROE', 4]}

    max_work_dt = maxworkdt_command()

    # 전종목 등락률 먼저 읽음
    selPeriods = ['Day', 'Week', 'Month', 'Quarter', 'Year']
    selPeriod = selPeriods[0]
    df_KRX_12002, strtDd = KRX_12002(max_work_dt, selPeriod)
    #display(df_KRX_12002)

    # 업종분류현황 그 다음 읽음
    df_KRX_12025 = KRX_12025(max_work_dt)
    # 숫자값이 없는 것으로 확인되면 re-try 하는 부분 보완
    mktTypes = ['KOSPI', 'KOSDAQ', 'ALL']
    # mktType = mktTypes[0]

    if mktType!='ALL':
        df_KRX_12025 = df_KRX_12025[(df_KRX_12025['MKT_TP_NM']==mktType)]

    # 두 개의 dataframe merge 처리
    df_KRX_12025 = pd.merge(df_KRX_12002, df_KRX_12025, how='inner', on='ISU_SRT_CD', suffixes=('_y', ''))


    # remove comma and change data type for MKKCAP and FLUC_RT column
    df_KRX_12025['MKTCAP'] = df_KRX_12025['MKTCAP'].str.replace('-', '0')
    df_KRX_12025['MKTCAP'] = df_KRX_12025['MKTCAP'].str.replace(',', '').astype(float)
    df_KRX_12025['FLUC_RT'] = df_KRX_12025['FLUC_RT_y'].str.replace(',', '')
    df_KRX_12025['FLUC_RT'] = np.where(df_KRX_12025['FLUC_RT']=='-', '', df_KRX_12025['FLUC_RT'])
    df_KRX_12025['FLUC_RT'] = df_KRX_12025['FLUC_RT'].str.strip()
    df_KRX_12025 = df_KRX_12025[(df_KRX_12025['FLUC_RT'] != '')]
    df_KRX_12025['FLUC_RT'] = df_KRX_12025['FLUC_RT'].astype(float)

    #df_KRX_12025['FLUC_RT_DAY'] = ((1 + df_KRX_12025['FLUC_RT']/100)**(1/dayBtn) - 1) * 100
    df_KRX_12025['FLUC_RT_DAY'] = ((df_KRX_12025['FLUC_RT'] - df_KRX_12025['FLUC_RT'].min()) / (df_KRX_12025['FLUC_RT'].max() - df_KRX_12025['FLUC_RT'].min()) * 25)
    df_KRX_12025['FLUC_RT_sig'] = 1 / (1 + np.exp(-df_KRX_12025['FLUC_RT']))


    # FLUC_AMT: 시총 기준 종목별 변동금액
    df_KRX_12025['FLUC_AMT'] = df_KRX_12025['MKTCAP'] * df_KRX_12025['FLUC_RT'] / 100
    # IND_FLUC_AMT: 업종별 변동금액
    df_KRX_12025['IND_FLUC_AMT'] = df_KRX_12025.groupby('IDX_IND_NM')['FLUC_AMT'].transform('sum')
    # IND_MKTCAP: 업종 시가총액
    df_KRX_12025['IND_MKTCAP'] = df_KRX_12025.groupby('IDX_IND_NM')['MKTCAP'].transform('sum')
    # UP_DN: 상승/하락 구분표기
    df_KRX_12025['UP_DN'] = np.where(df_KRX_12025['IND_FLUC_AMT']>=0, 'Up', 'Down')
    # IND_FLUC_RT: 업종 등락률
    df_KRX_12025['IND_FLUC_RT'] = df_KRX_12025['IND_FLUC_AMT'] / df_KRX_12025['IND_MKTCAP'] * 100

    # MKT_FLUC_AMT: 상승/하락별 변동금액
    df_KRX_12025['MKT_FLUC_AMT'] = df_KRX_12025.groupby('UP_DN')['FLUC_AMT'].transform('sum')
    # MKT_MKTCAP: 상승/하락별 시가총액
    df_KRX_12025['MKT_MKTCAP'] = df_KRX_12025.groupby('UP_DN')['MKTCAP'].transform('sum')
    # MKT_FLUC_RT: 상승/하락별 등락률
    df_KRX_12025['MKT_FLUC_RT'] = df_KRX_12025['MKT_FLUC_AMT'] / df_KRX_12025['MKT_MKTCAP'] * 100

    df_KRX_12025 = df_KRX_12025.replace(np.nan, '', regex=True)


    # 전종목 valuation
    # df_KRX_all_val = all_val(max_work_dt)
    # df_KRX_12025 = pd.merge(df_KRX_12025, df_KRX_all_val, how='left', on='ISU_SRT_CD', suffixes=('', '_z'))

    # df_KRX_12025['PER'] = df_KRX_12025['PER'].str.replace('-', '0')
    # df_KRX_12025['PER'] = df_KRX_12025['PER'].str.replace(',', '').astype(float)
    # df_KRX_12025['PER'] = df_KRX_12025['PER'].astype(float)
    # df_KRX_12025['PBR'] = df_KRX_12025['PBR'].str.replace('-', '0')
    # df_KRX_12025['PBR'] = df_KRX_12025['PBR'].str.replace(',', '').astype(float)
    # df_KRX_12025['PBR'] = df_KRX_12025['PBR'].astype(float)
    # df_KRX_12025['DVD_YLD'] = df_KRX_12025['DVD_YLD'].astype(float)
    # df_KRX_12025['ROE'] = df_KRX_12025['PBR'] / df_KRX_12025['PER'] * 100
    # df_KRX_12025['ROE'] = df_KRX_12025['ROE'].fillna(0)

    custom_color_scale = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'white', 'red', 'red', 'red', 'red', 'red', 'red', 'red']
    custom_color_scale = ['#30CC5A', '#2F9E4F', '#35764E', '#414554', '#8B444E', '#BF4045', '#F63538'] 

    selMetrics = ['Price', 'PER', 'PBR', 'Dividend', 'ROE']
    selMetric = selMetrics[0]
    color_range = {'Price':{"Day":[-3, 3], "Week":[-6, 6], "Month":[-10, 10], "Quarter":[-18, 18], "Year":[-30, 30]},
                'PER':{"Day":[0, 30], "Week":[0, 30], "Month":[0, 30], "Quarter":[0, 30], "Year":[0, 30]},
                'PBR':{"Day":[0, 5], "Week":[0, 5], "Month":[0, 5], "Quarter":[0, 5], "Year":[0, 5]},
                'Dividend':{"Day":[0, 5], "Week":[0, 5], "Month":[0, 5], "Quarter":[0, 5], "Year":[0, 5]},
                'ROE':{"Day":[0, 30], "Week":[0, 30], "Month":[0, 30], "Quarter":[0, 30], "Year":[0, 30]},
                }

    fig = px.treemap(df_KRX_12025, path=['MKT_TP_NM', 'IDX_IND_NM', 'ISU_ABBRV'], values='MKTCAP',
                    maxdepth=5,
                    # hover_data=['FLUC_RT', 'FLUC_AMT', 'IND_FLUC_RT', 'PER', 'PBR', 'DVD_YLD', 'ROE'],
                    hover_data=['FLUC_RT', 'FLUC_AMT', 'IND_FLUC_RT'],
                    color=dict_Metric[selMetric][0],
                    #color='IND_FLUC_RT',
                    #color_continuous_scale='Turbo',
                    color_continuous_scale= custom_color_scale,
                    #range_color=[-3, 3],
                    range_color=color_range[selMetric][selPeriod],
                    #color_continuous_midpoint=0,
                    width=1200,
                    height=700,
                    # custom_data=['IND_FLUC_RT', 'FLUC_RT', 'ISU_ABBRV', 'ISU_SRT_CD', 'MKT_TP_NM', 'MKT_FLUC_RT', 'PER', 'PBR', 'DVD_YLD', 'ROE'],
                    custom_data=['IND_FLUC_RT', 'FLUC_RT', 'ISU_ABBRV', 'ISU_SRT_CD', 'MKT_TP_NM', 'MKT_FLUC_RT'],
                    )

    # fig.data[0].customdata = np.column_stack([df_KRX_12025['FLUC_RT'].tolist(), df_KRX_12025['PER'].tolist(), df_KRX_12025['PBR'].tolist(), df_KRX_12025['DVD_YLD'].tolist(), df_KRX_12025['ROE'].tolist()])
    fig.data[0].customdata = np.column_stack([df_KRX_12025['FLUC_RT'].tolist()])

    if selMetric in ['Price', 'Dividend', 'ROE']:
        fig.data[0].texttemplate = "%{label}<br>%{customdata["+str(dict_Metric[selMetric][1])+"]:.2f}%"
    else:
        fig.data[0].texttemplate = "%{label}<br>%{customdata["+str(dict_Metric[selMetric][1])+"]:.1f}"


    fig.update_traces(hovertemplate=None, hoverinfo='skip', textfont_size=14)
    # fig.update_traces(hovertemplate='PER %{customdata[1]:.1f}<br>PBR %{customdata[2]:.1f}<br>Dividend %{customdata[3]:.1f}%<br>ROE %{customdata[4]:.1f}%', textfont_size=14)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
        autosize=False,
        width=1000,
        height=700,
    )

    return fig


with tab4:
    st.subheader("Treemap for Korean Stock Market")
    
    treemap_market = st.radio(label = 'Market', options = ['KOSPI', 'KOSDAQ', 'ALL'], index=2)
    st.write('<style>div.row-widget.stRadio> div{flex-direction:row;}</style>', unsafe_allow_html=True)

    ct = st.container()
    treemap_switch = ct.select_slider("Style", ["Simple", "Normal"]) 

    fig_4 = TreeMap(treemap_market)

    st.plotly_chart(fig_4, use_container_width=True)

    # pio.write_image(fig_4, "treemap.png")
    fig_4.write_image("treemap.png", engine='kaleido')
    with open("treemap.png", "rb") as file:
        btn = st.download_button(
                label="Download image",
                data=file,
                file_name="treemap.png",
                mime="image/png"
            )