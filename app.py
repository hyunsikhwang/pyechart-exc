from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import streamlit as st
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
from datetime import datetime
import requests
import bs4
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd


st.set_page_config(page_title="CNN Fear and Greed Index", layout="wide", page_icon="random")

st.header("pyechart")


url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

def get_bs(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.get(url, headers=headers).text, "lxml")


tab1, tab2 = st.tabs(["Fear and Greed Index", "Buffet Index"])

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
    strt_dd = (datetime.now() - relativedelta(years=15)).strftime("%Y%m%d")

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
    data = json.loads(MktData.text)
 
    elevations = json.dumps(data['output'])
    day_one = pd.read_json(elevations)
    org_df = pd.DataFrame(day_one)
 
    return org_df


ecos_api_key = st.secrets["ecos_api_key"]

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


df_idx_KOSPI = idx_prc('KOSPI')
df_idx_KOSPI['MKTCAP'] = df_idx_KOSPI['MKTCAP'].replace({',':''}, regex=True).astype(float)
df_idx_KOSDAQ = idx_prc('KOSDAQ')
df_idx_KOSDAQ['MKTCAP'] = df_idx_KOSDAQ['MKTCAP'].replace({',':''}, regex=True).astype(float)

df_idx_prc = df_idx_KOSPI.merge(df_idx_KOSDAQ, how='left', left_on='TRD_DD', right_on='TRD_DD')
df_idx_prc['MKTCAP'] = df_idx_prc['MKTCAP_x'] + df_idx_prc['MKTCAP_y']

df_idx_prc['TRD_DD'] = pd.to_datetime(df_idx_prc['TRD_DD']).dt.strftime('%Y-%m-%d')
df_idx = df_idx_prc[['TRD_DD', 'MKTCAP_x', 'MKTCAP', 'CLSPRC_IDX_x']].copy()
df_idx.rename({'MKTCAP_x':'MKTCAP_KOSPI', 'CLSPRC_IDX_x':'CLSPRC_IDX'}, axis='columns', inplace=True)


df_new = pd.merge(df_idx, df, how='left', left_on='TRD_DD', right_on='TIME')
#display(df_new.dtypes)
df_new['Ratio'] = df_new['MKTCAP'] / df_new['AnnSum']
df_new['Ratio_KOSPI'] = df_new['MKTCAP_KOSPI'] / df_new['AnnSum']



df_new['TRD_DD'] = pd.to_datetime(df_new['TRD_DD'], format="%Y-%m-%d")

#df_new = df_new.set_index('TIME').resample('D').interpolate(method='cubic').reset_index()
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
initRatio = df_new['Ratio_KOSPI'].median()
maxRatio = df_new['Ratio_KOSPI'].max()

Ratio_max = min(df_new['Ratio_KOSPI'].max()*0.9, initRatio * 3)
Ratio_min = max(df_new['Ratio_KOSPI'].min()*1.2, initRatio * 0.6)
Ratio_high = (initRatio + Ratio_max) * 0.5
Ratio_low = (initRatio + Ratio_min) * 0.5
band = f'{initRatio:.0%} base'

# Buffet Index Range
Ratio_max = 1.08
Ratio_min = 0.58
Ratio_high = 0.91
Ratio_low = 0.75

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

df_new = df_new.fillna(method='ffill')

x_data = list(df_new['TRD_DD'].dt.strftime("%Y-%m-%d"))
y_data = list(df_new['MKTCAP_KOSPI'])
y_data_ratio = list(df_new['Ratio_KOSPI'])
y_max = df_new['MKTCAP_KOSPI'].max()*1.1

y_bandMin = list(df_new[bandMin])
y_bandLow = list(df_new[bandLow]-df_new[bandMin])
y_bandHigh = list(df_new[bandHigh]-df_new[bandLow])
y_bandMax = list(df_new[bandMax]-df_new[bandHigh])
y_bandLimit = list(y_max-df_new[bandMax])

if y_data[-1] >= Ratio_max:
    y_status = 'Significantly Overvalued'
elif y_data[-1] >= Ratio_high:
    y_status = 'Modestly Overvalued'
elif y_data[-1] >= Ratio_low:
    y_status = 'Fair Valued'
elif y_data[-1] >= Ratio_min:
    y_status = 'Modestly Undervalued'
else:
    y_status = 'Significantly Undervalued'


with tab2:
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
                data=[opts.MarkPointItem(name="Current", type_=None, coord=[x_data[-1], y_data[-1]], value=f"{y_data_ratio[-1]:.2f}\n\n{y_status}")]),
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
