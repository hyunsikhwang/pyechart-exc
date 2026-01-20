from pyecharts import options as opts
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

    # Pivot for pyecharts
    df_pivot = df_tot.pivot(index='TIME', columns='ITEM_NAME1', values='DATA_VALUE')
    x_axis = df_pivot.index.strftime('%Y-%m-%d').tolist()

    line = Line(init_opts=opts.InitOpts(width="100%", height="600px"))
    line.add_xaxis(x_axis)

    for column in df_pivot.columns:
        line.add_yaxis(
            series_name=column,
            y_axis=df_pivot[column].tolist(),
            is_smooth=True,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Bond Yield"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(type_="value", is_scale=True),
        datazoom_opts=[opts.DataZoomOpts(type_="slider", range_start=80, range_end=100)],
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )

    # Render using components.html to match the pattern in tab1 if st_pyecharts has issues
    components.html(line.render_embed(), height=600)
