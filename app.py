import streamlit as st
import pandas as pd
import re
import plotly.express as px


# read data in df
@st.cache_data
def getdata(path):
    return pd.read_csv(path, header = None, names=["date", "linac", "file_name", "item", 'part', "value"], )

# filter df on a certain value in the col
def select_data(filter_on, df, col_n):
    df[col_n] = df[col_n].astype(str)
    return df[df[col_n].str.contains(fr'{str(filter_on)}', regex=True)]

# make the total amount of changes fig
def total_amount_fig(df):
    selected_df = select_data(st.session_state.type_file_total_fig, df, col_n="file_name")

    changes_per_file = make_changes_per_file(selected_df)
    plotting_per_file = make_total_amount_to_plot(changes_per_file)

    fig = px.line(plotting_per_file, x='Date', y='Amount', color='File name', markers=True, symbol='Linac', hover_data="First entry", title= "Total amount of differences per file per linac")
    
    return fig

@st.cache_data #??? not sure, want wordt meeerdere malen aangeroepen
def make_total_amount_to_plot(changes_per_file):
    #get the sum of the amount of changes
    plotting_per_file = pd.DataFrame()
    for file_name, df_changes in changes_per_file.items():
        for linac in set(df_changes["linac"]):
            # get how many times there was a diffence between col
            date_columns = df_changes.columns.difference(['date', 'linac', 'file_name', 'item', 'part', df_changes.columns[-1]])
            plotting = list((df_changes[date_columns] != 0).sum()[:-1]) 

            first_date = re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-1])[0]

            # # set it in the dict
            data = {
                "Amount": plotting,
                "Date": list(date_columns[:-1]),
                "Linac": [linac] * len(plotting),
                "File name": [file_name] * len(plotting),
                "First entry": [first_date] * len(plotting)
            }
            plotting_per = pd.DataFrame(data) 
            plotting_per_file = pd.concat([plotting_per_file, plotting_per])
    return plotting_per_file

@st.cache_data #??? not sure, want wordt meeerdere malen aangeroepen
def make_changes_per_file(df):
    # change df
    df_pivot = df.pivot(index=["linac", "file_name", "item", "part"], columns="date", values="value")
    df_pivot.reset_index(inplace=True)

    # the dates all the different file types been uploaded
    file_dates = {name : sorted(list(set(df[df["file_name"] == name]["date"]))) for name in set(df["file_name"])}

    #how much every part&item changed
    changes_per_file = {}

    for file_name, file_date in file_dates.items():
        # grab the dates that this file has been uploaded
        df_change = df_pivot[df_pivot["file_name"] == file_name].loc[:,file_date]
        # Calculate how much difference there is between old and new
        df_changes =  df_change.iloc[:,1:].values - df_change.iloc[:,:-1].values
        # make a df from the changes and the information of linac, item en part
        df_changed = pd.concat([df_pivot[df_pivot["file_name"] == file_name].loc[:,["linac", "item", "part"]].reset_index(), 
                pd.DataFrame(df_changes, columns=df_change.columns[1:])], axis=1)
        # add first entry date
        df_changed[f"first_entry: {file_date[0]}"] = df_change.iloc[:,0].values
                    
        changes_per_file[file_name] = df_changed

    return changes_per_file

## to make the bulk of the data smaller
@st.cache_data #??? not sure, want wordt meeerdere malen aangeroepen
def select_linac(df):
    return select_data(st.session_state.linac_linac_compare_df1, df, col_n="linac")

def make_compare_df(df):
    # grab only selected data
    newdf = select_linac(df)

    # combine the 2 dates
    filterddf = pd.concat(
        [select_data(st.session_state.old_date_linac_compare_df1, newdf, col_n="date"), 
        select_data(st.session_state.new_date_linac_compare_df1, newdf, col_n="date")]
    )

    # calculate the difference
    changes_per_file = make_changes_per_file(filterddf)
    
    #to remove the item & part if has 0 difference
    if st.session_state.none_changes_compare_linacs == False:
        changes_per_file = {filename: df_changes[(df_changes.iloc[:,4:-1] != 0).any(axis=1)] for filename, df_changes in changes_per_file.items()}
    #make it to a df to show
    total_df = pd.DataFrame()
    for filename, df_changes in changes_per_file.items():
        if df_changes.columns[-2] != "part":

            old_data = df_changes[df_changes.columns[-1]]
            dif = df_changes[df_changes.columns[-2]]

            data = {
                    "File name": [filename] * len(df_changes),
                    "Item & part": "I:"+ df_changes["item"].astype(str) + " P:"+ df_changes["part"].astype(str),
                    "Difference": list(dif),
                    "Old data": list(old_data),
                    "New data": list(old_data + dif)
                }
            temp_df = pd.DataFrame(data) 
            total_df = pd.concat([total_df, temp_df])

    
    return total_df

@st.cache_data #??? not sure, want wordt meeerdere malen aangeroepen
def select_file_name(df):
        return select_data(st.session_state.file_type_diff_plot, df, "file_name")


def make_diff_plot_comparedf(df):
    newdf = select_file_name(df) 
    changes_per_file = make_changes_per_file(newdf)

    total_df = pd.DataFrame()
    ips = st.session_state.i_and_p
    for filename, df_changes in changes_per_file.items():
        if df_changes.columns[-2] != "part":
            #add of the item
            df_changes["Item & part"] = "I:"+ df_changes["item"].astype(str) + " P:"+ df_changes["part"].astype(str)
            dates = list(df_changes.columns)[4:-2]
            for ip in ips:
                ip_df = df_changes[df_changes["Item & part"] == ip]

                data = {
                        "Linac": [df_changes["linac"][1]] * len(dates),
                        "Item & part": [ip] * len(dates),
                        "date": dates,
                        "Difference": ip_df[dates].values[0],
                        "First entry": ip_df.iloc[:,-2].values[0]
                    }
                
                temp_df = pd.DataFrame(data) 
                total_df = pd.concat([total_df, temp_df])
    fig = px.line(total_df, x='date', y='Difference', color='Item & part', markers=True, symbol='Linac', hover_data="First entry", title= "Differences per Item & part per linac over the time")
    return fig

def add_compare_linac(layout):
    return



df_db = getdata("data/cal_changed.csv")

# layout
layout = st.container(border=True)
layout.title("Linac logfiles comparing")

amount_plot = st.container(border= True)
amount_plot.selectbox(label="The type of file you want to see", options=["Be52", "Be63", "Optics", "Mlc"], key="type_file_total_fig")
amount_plot.plotly_chart(total_amount_fig(df_db))

compare_linacs = st.container(border=True)
cl1, cl2, cl3 = compare_linacs.columns(3)
cl1.checkbox("See the Part & items that had no changes", key="none_changes_compare_linacs")
multiple1, multiple2 = compare_linacs.columns(2)
compare_linac = multiple1.container()
compare_linac.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df1")
cl2.selectbox(label="The date you want to compare to", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"]))), key="old_date_linac_compare_df1")
cl3.selectbox(label="The date you want to see the difference of", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"])))[1:], key="new_date_linac_compare_df1")
compare_df = compare_linac.data_editor(make_compare_df(df_db), key="linac_compare_df")
# multiple2.button(icon = ":material/add:", help= "add",label="", on_click=add_compare_linac, args=compare_linacs)
compare_linac2 = multiple2.container()
compare_linac2.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df2")
compare_df2 = compare_linac2.data_editor(make_compare_df(df_db), key="linac_compare_df2")


diff_plot = st.container(border=True)
btn1, btn2 = diff_plot.columns(2)
btn1.selectbox(label='File type', options=["Be5201"], key="file_type_diff_plot")
btn2options = list(compare_df[compare_df["File name"] == st.session_state.file_type_diff_plot]["Item & part"].values)
try:
    btn2.multiselect(label="Part & item combination", options=btn2options, default=btn2options[0], key="i_and_p")
    diff_plot.plotly_chart(make_diff_plot_comparedf(df_db))
except IndexError:
    diff_plot.write("Select the dates in the table above that the file is twice present")

