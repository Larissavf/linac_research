import streamlit as st
import pandas as pd
import re
import plotly.express as px
from itertools import accumulate
from plotly.subplots import make_subplots
import plotly.graph_objects as go

"""
read data from the database in df

input:
    path, string: path to the file
return:
    dataframe
"""
@st.cache_data
def get_db_data(path):
    return pd.read_csv(path, header = None, names=["date", "linac", "file_name", "item", 'part', "value"])

"""
read data from different translate tables in df

input:
    path, string: path to the file
return:
    dataframe
"""
@st.cache_data
def get_transl_data(path):
    return pd.read_csv(path)

"""
read data in a tsv form

input:
    path, string: path to the file
return:
    dataframe
"""
@st.cache_data
def get_grouped_data(path):
    return pd.read_csv(path, sep="\t")


"""
filter df on a certain value in the col

input:
    filter_on, string: value it needs to be filterd on
    df, dataframe: object that will be filterd
    col_n, string: name of the column in the df
return:
    filterd dataframe
"""
def select_data(filter_on, df, col_n):
    df[col_n] = df[col_n].astype(str)
    return df[df[col_n].str.contains(fr'{str(filter_on)}', regex=True)]

"""
make the total amount of changes figure

input:
    df, dataframe: object that will be filterd
return:
    plotly line object
"""
def total_amount_fig(df):
    selected_df = select_data(st.session_state.linac_total_fig, df , col_n="linac")
    selected_df = select_data(st.session_state.type_file_total_fig, selected_df, col_n="file_name")

    changes_per_file = make_changes_per_file(selected_df)
    plotting_per_file = make_total_amount_to_plot(changes_per_file, st.session_state.see_part_45)
    st.session_state["changes_per_file"] = changes_per_file
    st.session_state["types_file_total_amount"] = sorted(list(set(plotting_per_file["File name"])))
    fig_total = px.line(plotting_per_file, x='Date', y='Amount', color='File name', markers=True, hover_data="First entry", title= "Total amount of differences per file per linac")
    return fig_total



"""
make the object for the total mount plot

input:
    changes_per_file, dict: contains filenames, df_changes: df with the changes from last date 
return:
    plotting_per_file, dataframe for plotting the total amount plot
"""
@st.cache_data 
def make_total_amount_to_plot(changes_per_file, see_part_45):
    #get the sum of the amount of changes
    plotting_per_file = pd.DataFrame()
    for file_name, df_changes in changes_per_file.items():
            # get how many times there was a diffence between col
            date_columns = df_changes.columns.difference(['date', 'linac', 'file_name', 'item', 'part', df_changes.columns[-1]])

            #to not see part 45
            if see_part_45 == False:
                df_changes = df_changes[df_changes["part"] != 45]

            plotting = list((df_changes[date_columns] != 0).sum()[:-1]) 

            first_date = re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-1])[0]

            # # set it in the dict
            data = {
                "Amount": plotting,
                "Date": list(date_columns[:-1]),
                "File name": [file_name] * len(plotting),
                "First entry": [first_date] * len(plotting)
            }
            plotting_per = pd.DataFrame(data) 
            plotting_per_file = pd.concat([plotting_per_file, plotting_per])


    return plotting_per_file

"""
make the changes_per_file object
contains per file the changes it has made since the last date

input:
    df, dataframe
return:
    changes_per_file, dict: contains filenames, df_changes: df with the changes from last date 
"""
@st.cache_data 
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

"""
Make the data for the grouped amount plot

input:
    changes_per_file: dict, containing the changes per date per file
    groups: dataframe, the groups by every item
    file_name: string, containing the file that needs to be shown
return:
    plotly figure
"""
@st.cache_data 
def grouped_plot(changes_per_file, groups, file_name):
    #get the correct df_changes
    df = changes_per_file[file_name]

    #get amount of changes per date
    date_columns = df.columns.difference(['date', 'linac', 'file_name', 'item', 'part', df.columns[-1]])
    #add the correct group after the correct item
    grouped = pd.merge(df, groups[["item","groep"]], on="item", how="left")
    
    #make the df for plotting of this data
    plotting_per_group = pd.DataFrame()
    for group in set(grouped["groep"]):
        df_changes = grouped[grouped["groep"] == group]
        #get de sum of amount of changes
        plotting = list((df_changes[date_columns] != 0).sum()[:-1]) 
        first_date = re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-3])[0]

        data = {
            "Amount" : plotting,
            "Date": list(date_columns[:-1]),    
            "File name": [file_name] * len(plotting),
            "First entry": [first_date] * len(plotting),
            "Group": [group] * len(plotting)
        }

        plotting_per = pd.DataFrame(data) 
        plotting_per_group = pd.concat([plotting_per_group, plotting_per])
    fig = make_mulitple_grouped_plot(plotting_per_group, split_on="Group", plot_on="Amount")
    fig.update_layout(title_text="Grouped amount of differences per file type", height=150*len(list(set(plotting_per_group["Group"]))))

    return fig

"""
Make the fig for the grouped amount plot

input:
    dataframe: with the data connected to the groups to be plotted
    split_on: string, col name that it will be grouped on
    plot_on: string, col name that will be plotted
return:
    plotly figure
"""
@st.cache_data 
def make_mulitple_grouped_plot(plotting, split_on, plot_on):
    #get the differnt types of groups
    groups = list(set(plotting[split_on]))
    groups = [x for x in groups if str(x) != 'nan']

    #make a subplot for every type of group
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, subplot_titles=groups)

    for i, group in enumerate(groups):
        df = plotting[plotting[split_on] == group]
        fig.add_trace(
            go.Scatter(
                x=df["Date"], 
                y=df[plot_on], 
                mode="lines+markers",
                name=group,
                showlegend=False,
            ),
            row=i+1, col=1
        )
    groups

    return fig
    

"""
to make the bulk of the data smaller for compare df

input:
    df, dataframe
return:
    selected dataframe 
"""
@st.cache_data 
def select_linac(df):
    return select_data(st.session_state.linac_linac_compare_df1, df, col_n="linac")

"""
make the dataframe to display that contains the differences per item per date

 {
                    "File name": 
                    "Item & part": 
                    "Difference": 
                    "Old data": 
                    "New data": 
                }


input:
    df, dataframe
return:
    dataframe 
"""
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
                    "Difference": dif.tolist(),
                    "Old data": old_data.tolist(),
                    "New data": list(old_data + dif)
                }

            temp_df = pd.DataFrame(data) 
            total_df = pd.concat([total_df, temp_df])

    
    return total_df

"""
to make the bulk of the data smaller for compare df

input:
    df, dataframe
return:
    selected dataframe 
"""
@st.cache_data 
def select_file_name(df, filter):
        return select_data(filter, df, "file_name")

def multiple_plot_nes(df):
    maxus = []
    #get the type of different kind is present
    items = list(set(df["Item & part"]))
    for item in items:
        maxus.append(int(df[df["Item & part"] == item]["Value"].max()))
    # determine the differences between the lowest and highest value of a kind
    s_maxus = sorted(maxus)
    if s_maxus[0] - s_maxus[-1] > 2000 or s_maxus[0] - s_maxus[-1] < -2000:
        return True
    return False

"""
make the plot to compare multiple part and items

input:
    df, dataframe: 
return:
    plotly line object
"""
@st.cache_data 
def make_diff_plot_comparedf(df, ips, filetype):
    newdf = select_file_name(df, filetype) 

    # make the differences per file
    changes_per_file = make_changes_per_file(newdf)

    #make the df to show
    total_df = pd.DataFrame()
    for filename, df_changes in changes_per_file.items():
        if df_changes.columns[-2] != "part":
            #add of the item en part
            df_changes["Item & part"] = "I:"+ df_changes["item"].astype(str) + " P:"+ df_changes["part"].astype(str)

            dates = list(df_changes.columns)[4:-2]
            dates.append(re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-2])[0])
            dates = sorted(dates)


            for ip in ips:
                ip_df = df_changes[df_changes["Item & part"] == ip]

                #make the amount values again from the differences
                first_entry = ip_df.iloc[:,-2].values[0]
                difference = ip_df[dates[1:]].values[0]
                total_list = list(accumulate([first_entry] + difference.tolist()))

                data = {
                        "Linac": [df_changes["linac"][1]] * len(dates),
                        "Item & part": [ip] * len(dates),
                        "Date": dates,
                        "Value": total_list
                    }
                
                temp_df = pd.DataFrame(data) 
                total_df = pd.concat([total_df, temp_df])
    if multiple_plot_nes(total_df):
        fig = make_mulitple_grouped_plot(total_df, split_on="Item & part", plot_on="Value")
        fig.update_layout(title_text="The value per Item & part per linac over the time", height=150*len(list(set(total_df["Item & part"]))))
    else:
        fig = px.line(total_df, x='Date', y='Value', color='Item & part', markers=True, symbol='Linac', title= "The value per Item & part per linac over the time")
    return fig


@st.cache_data 
def translate_filename(df, translate):
    #get the translation for part 32
    trans = {filenam: translate[value] for value, filenam in df[df["item"] == 32][["value", "file_name"]].values if value <= 19 and value > 0}

    # apply the translation
    for filename, value in trans.items():
        df = df.replace(filename, value)
    return df

# Function to check if a value is numeric (either integer or float)
def is_numeric(value):
    try:
        # Try to convert to a float
        float(value[0])
        return True
    except :
        return False
    
"""
The value need to be transformed with a certain value, based on the elekta manual.

input: 
    dataframe, that needs to be translated
return:
    dataframe, that has been adjusted to the certain value for a item
"""
def translate_values(df):
    # get translate tabel
    linac_items = st.session_state.linac_items
    # change the strings to numbers
    resolution = linac_items[linac_items["Resolution"].apply(is_numeric)]
    # merge that the resolution numbers are behind the proper items
    df = df.merge(resolution[["Item name", "Resolution"]], left_on="item", right_on="Item name", how="left").drop(columns=["Item name"])

    # the values of the items times the resolution, that the value is of proper value
    mask = df["Resolution"].notna()
    res_split = df.loc[mask, "Resolution"].str.split(" ", expand=True)
    df.loc[mask, "value"] = df.loc[mask, "value"].astype(float) * res_split[0].astype(float)
    df.loc[mask, "Resolution"] = res_split[1]
    
    return df

@st.cache_data 
def edit_db_data(df):
    # to translate energy filename
    energy_dict = {
        0: "No energy set",
        1: "Low X-ray energy - 6 MV",
        2: "High X-ray energy - 25 MV",
        3: "energy: Radiographic X-ray",
        4: "Third X-ray energy - 10 MV",
        5: "6 MV FFF energy",
        6: "10 MV FFF energy",
        7: "X MV FFF energy - Not used MV",
        8: "energy: HDRE 1",
        9: "energy: HDRE 2",
        11: "Electron energy 1 - 4 MeV",
        12: "Electron energy 2 - 6 MeV",
        13: "Electron energy 3 - 8 MeV",
        14: "Electron energy 4 - 10 MeV",
        15: "Electron energy 5 - 12 MeV",
        16: "Electron energy 6 - 15 MeV",
        17: "Electron energy 7 - 18 MeV",
        18: "Electron energy 8 - 20 MeV",
        19: "Electron energy 9 - 22 MeV"
    }
    #translate the energyfile names
    df = translate_filename(df, energy_dict)

    df = translate_values(df)

    return df


df_db = get_db_data("data/cal_changed.csv")

linac_items = get_transl_data("data/translate_tbl.csv")
linac_items["Item name"] =  linac_items["Item name"].str.extract(r'i(\d+)').astype("float")
st.session_state["linac_items"] = linac_items

agility_parts_items = get_transl_data("data/agility_parts_items.csv")
st.session_state["agility_parts_items"] = agility_parts_items


linac_parts = get_transl_data("data/Itemparts.csv")

grouped_partly = get_grouped_data("data/groups_partly.tsv")

df_db = edit_db_data(df_db)

# layout
layout = st.container(border=True)
layout.title("Linac logfiles comparing")

amount_plot = st.container(border= True)
a1, a2, a3 = amount_plot.columns(3)
a1.checkbox("See part 45", key="see_part_45")
a2.selectbox(label="The type of file you want to see", options=["energy", "Be63", "Optics", "Mlc"], key="type_file_total_fig")
a3.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_total_fig")
amount_plot.plotly_chart(total_amount_fig(df_db))
amount_plot.selectbox(label="The type of file you want to see", options=st.session_state.types_file_total_amount, key="type_file_grouped_fig")
grouped_amount_plot = amount_plot.expander(label="See the changes per group")
grouped_amount_plot.plotly_chart(grouped_plot(st.session_state.changes_per_file, grouped_partly, st.session_state.type_file_grouped_fig))

compare_linacs = st.container(border=True)
cl1, cl2, cl3 = compare_linacs.columns(3)
cl1.checkbox("See the Part & items that had no changes", key="none_changes_compare_linacs")

multiple1, multiple2  = compare_linacs.columns(2)
compare_dfs = compare_linacs.expander(label= "see the dataframes with the selected data")
df1, df2  = compare_dfs.columns(2)


multiple1.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df1")
cl2.selectbox(label="The date you want to compare to", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"]))), key="old_date_linac_compare_df1")
cl3.selectbox(label="The date you want to see the difference of", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"])))[1:], key="new_date_linac_compare_df1")
compare_df = df1.data_editor(make_compare_df(df_db), key="linac_compare_df")
multiple2.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df2")
compare_df2 = df2.data_editor(make_compare_df(df_db), key="linac_compare_df2")


diff_plot = st.container(border=True)
btn1, btn2 = diff_plot.columns(2)
btn1.selectbox(label='File type', options=set(compare_df["File name"]), key="file_type_diff_plot")
btn2options = list(compare_df[compare_df["File name"] == st.session_state.file_type_diff_plot]["Item & part"].values)
try:
    btn2.multiselect(label="Part & item combination", options=btn2options, default=btn2options[0], key="i_and_p")
    diff_plot.plotly_chart(make_diff_plot_comparedf(df_db, st.session_state.i_and_p ,st.session_state.file_type_diff_plot))
except Exception as e:
    print(e)
    diff_plot.write("Select the dates in the table above that the file is twice present and then choose a Part & item you want to see")

