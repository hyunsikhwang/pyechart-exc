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


st.set_page_config(page_title="CNN Fear and Greed Index", layout="wide", page_icon="random")

st.header("pyechart")


url = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'

def get_bs(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    return bs4.BeautifulSoup(requests.get(url, headers=headers).text, "lxml")

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
                opts.MarkAreaItem(name="EXTREME FEAR", y=(0, 25), itemstyle_opts=opts.ItemStyleOpts(color="red", opacity=0.5)),
                opts.MarkAreaItem(name="FEAR", y=(25, 45), itemstyle_opts=opts.ItemStyleOpts(color="orange", opacity=0.5)),
                opts.MarkAreaItem(name="NEUTRAL", y=(45, 55), itemstyle_opts=opts.ItemStyleOpts(color="yellow", opacity=0.5)),
                opts.MarkAreaItem(name="GREED", y=(55, 75), itemstyle_opts=opts.ItemStyleOpts(color="green", opacity=0.5)),
                opts.MarkAreaItem(name="EXTREME GREED", y=(75, 100), itemstyle_opts=opts.ItemStyleOpts(color="blue", opacity=0.5)),
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
