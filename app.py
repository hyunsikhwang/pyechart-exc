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
from datetime import datetime, timedelta
from pykrx import stock
from pytz import timezone, utc
import plotly.express as px
import os
import plotly.graph_objects as go

st.set_page_config(page_title="CNN Fear and Greed Index", layout="wide", page_icon="random")

st.header("pyechart")

url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

def get_bs(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.get(url, headers=headers).text, "lxml")

tab1, tab5, tab6 = st.tabs(["Fear and Greed Index", "Waterfall", "Bond Yield"])

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
    components.html(line, height=800)

# KOSPI Waterfall
with tab5:
    KST = timezone('Asia/Seoul')
    now = datetime.utcnow()
    SeoulTime = utc.localize(now).astimezone(KST).strftime('%Y%m%d')

    # Multi-ticker and multi-source fallback strategy
    tickers = ['1001', '코스피', 'KOSPI']
    df = None
    last_error = ""

    for ticker in tickers:
        try:
            # Try fundamental first
            df = stock.get_index_fundamental('20020101', SeoulTime, ticker)
            if df is not None and not df.empty:
                break
            # Try ohlcv if fundamental is empty
            df = stock.get_index_ohlcv_by_date('20020101', SeoulTime, ticker)
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_error = str(e)
            continue
    
    if df is not None and not df.empty:
        df = df.reset_index()
        # Positional indexing for name-agnostic stability
        if len(df.columns) >= 2:
            df1 = df.iloc[:, [0, 1]].copy()
            df1.columns = ['날짜', '종가']
            df2 = df1[~df1['날짜'].dt.strftime('%Y-%m').duplicated()].copy()
        else:
            st.error(f"데이터 구조가 올바르지 않습니다. (Columns: {df.columns.tolist()})")
            st.stop()
    else:
        st.warning(f"KOSPI 데이터를 수집할 수 없습니다. (Tickers tried: {tickers})")
        if last_error:
            st.info(f"상세 오류: {last_error}")
        st.stop()

    df2 = pd.concat([df2, df1.tail(1)])
    df2['pct'] = df2['종가'].pct_change(periods=1, axis=0)
    df2['diff'] = df2['종가'].diff()

    fig = go.Figure(go.Waterfall(
        name = "KOSPI", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative", ],
        x = df2['날짜'],
        textposition = "outside",
        text = df2['종가'],
        y = df2['diff'],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_xaxes(dtick="M12")
    fig.update_xaxes(showgrid=True, minor_showgrid=True, gridwidth=1, griddash='dash', gridcolor='LightPink')

    st.plotly_chart(fig, use_container_width=True)

# Bond Yield
with tab6:
    KST = timezone('Asia/Seoul')
    now = datetime.utcnow()
    SeoulTime = utc.localize(now).astimezone(KST)
    nowSeo = SeoulTime.strftime('%Y%m%d')

    bond_cd = {'0101000': '722Y001',
               '010190000': '817Y002',
               '010200000': '817Y002',
               '010210000': '817Y002',
               '010220000': '817Y002',
               '010230000': '817Y002',
               '010240000': '817Y002',
               '010300000': '817Y002',
               }

    df_tot = pd.DataFrame()

    #금리
    for (bondcd, bondcd1) in zip(list(bond_cd.values()), list(bond_cd.keys())):
        url = f'http://ecos.bok.or.kr/api/StatisticSearch/967SFAC1NLQO1Z31HUMX/json/kr/1/10000/{bondcd}/D/20020101/{nowSeo}/{bondcd1}'

        res = requests.get(url)
        resJsn = json.loads(res.text)['StatisticSearch']['row']

        df = pd.DataFrame(resJsn)
        df['DATA_VALUE'] = df['DATA_VALUE'].astype(float)
        df['TIME'] = pd.to_datetime(df['TIME'])

        df_tot = pd.concat([df_tot, df])

    fig = px.line(df_tot, x='TIME', y='DATA_VALUE', color='ITEM_NAME1')
    fig.update_xaxes(dtick='M12', showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across", spikethickness=1)
    fig.update_xaxes(showgrid=True, minor_showgrid=True, gridwidth=1, griddash='dash', gridcolor='LightPink')
    fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=1)

    st.plotly_chart(fig, use_container_width=True)

