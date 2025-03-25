import streamlit as st
import pandas as pd
import plotly.express as px



@st.cache_data
def getdata(path):
    return pd.read_csv(path, header = None, names=["date", "linac", "file_name", "item", 'part', "value"], )

@st.cache_data
def select_data(filter_on, df, col_n):
    return df[df[col_n].str.contains(fr'{filter_on}', regex=True)]

def total_amount_fig(df):
    selected_df = select_data(st.session_state.type_file_total_fig, df, col_n="file_name")

    file_names = set(selected_df["file_name"])

    



df_db = getdata("data/cal_changed.csv")

fig_total = px.line(total_amount_fig(df_db), x='', y='lifeExp', color='country', markers=True)

# layout
layout = st.container()
layout.title("Linac vergelijken")
layout.selectbox(label="The type of file you want to see", options="Be52", key="type_file_total_fig")
# layout.plotly_chart(fig_total)