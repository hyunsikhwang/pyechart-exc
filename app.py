from pyecharts import options as opts
from pyecharts.charts import TreeMap
from streamlit_echarts import st_pyecharts
import streamlit as st


st.header("pyechart treemap sample")

data = [
    {
        "value": 40,
        "name": "클래스 A",
    },
    {
        "value": 180,
        "name": "클래스 B",
        "children": [
            {
                "value": 76,
                "name": "B",
                "children": [
                    {
                        "value": 12,
                        "name": "b-a",
                    },
                    {
                        "value": 28,
                        "name": "b-b",
                    },
                    {
                        "value": 20,
                        "name": "b-c",
                    },
                    {
                        "value": 16,
                        "name": "b-d",
                    }]
            }]}
]

treemap = TreeMap()
treemap.add("이건뭐지", data)
st_pyecharts(treemap)
