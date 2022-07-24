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
    Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
    .add_xaxis(x_axis)
    .add_yaxis('Index',
               y_axis,
               is_smooth=True,
               is_step=False,
               label_opts=opts.LabelOpts(
                    formatter=JsCode(
                        "function(params){return params.value[1].toFixed(1);}"
                    )
               ),
               markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="CNN Fear and Greed Index"),
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                "function (params) {return params.value[0] + '<br>' + params.value[1].toFixed(1);}"
            )
        ),
        xaxis_opts=opts.AxisOpts(interval=0)
    )
    .render_embed()
)
#line.render()


#st_pyecharts(line, width=1000, height=800)
components.html(line)
