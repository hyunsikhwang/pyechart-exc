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
from pytz import timezone, utc
import plotly.express as px
from db_handler import BondDBHandler

st.set_page_config(page_title="CNN Fear and Greed Index", layout="wide", page_icon="random")

st.header("pyechart")

url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

def get_bs(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.get(url, headers=headers).text, "lxml")

tab1, tab6 = st.tabs(["Fear and Greed Index", "Bond Yield"])

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

    db = BondDBHandler()

    #금리
    for (bondcd, bondcd1) in zip(list(bond_cd.values()), list(bond_cd.keys())):
        last_date = db.get_last_date(bondcd, bondcd1)

        if last_date:
            start_date_obj = datetime.strptime(last_date, '%Y%m%d') + timedelta(days=1)
            start_date_str = start_date_obj.strftime('%Y%m%d')
        else:
            start_date_str = '20020101'

        if start_date_str <= nowSeo:
            url = f'http://ecos.bok.or.kr/api/StatisticSearch/967SFAC1NLQO1Z31HUMX/json/kr/1/10000/{bondcd}/D/{start_date_str}/{nowSeo}/{bondcd1}'

            try:
                res = requests.get(url)
                data = json.loads(res.text)
                if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                    resJsn = data['StatisticSearch']['row']
                    df = pd.DataFrame(resJsn)
                    db.save_data(df)
            except Exception as e:
                print(f"Error fetching data for {bondcd}/{bondcd1}: {e}")

    # Load all data from DB
    df_tot = db.get_all_data(None, None)

    if not df_tot.empty:
        df_tot['DATA_VALUE'] = df_tot['DATA_VALUE'].astype(float)
        df_tot['TIME'] = pd.to_datetime(df_tot['TIME'])

        # Ensure data is sorted
        df_tot = df_tot.sort_values(by='TIME')

    # Prepare data for Pyecharts
    unique_dates = sorted(df_tot['TIME'].unique()) if not df_tot.empty else []
    x_axis = [d.strftime('%Y-%m-%d') for d in unique_dates]

    line = Line(init_opts=opts.InitOpts(width="100%", height="600px"))
    line.add_xaxis(x_axis)

    for bond_name in df_tot['ITEM_NAME1'].unique():
        bond_data = df_tot[df_tot['ITEM_NAME1'] == bond_name]
        data_map = {row['TIME'].strftime('%Y-%m-%d'): row['DATA_VALUE'] for _, row in bond_data.iterrows()}
        y_values = [data_map.get(date_str, None) for date_str in x_axis]

        line.add_yaxis(
            series_name=bond_name,
            y_axis=y_values,
            is_smooth=True,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            is_connect_nones=True
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Bond Yields"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(type_="value", min_='dataMin'),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )

    st_pyecharts(line, height="600px")
