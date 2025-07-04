import streamlit as st
import numpy as np
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
def get_db_data(headerpath, itempath, valuespath):
    #load in the different files
    headers = pd.read_csv(headerpath, header = None,  names=["date", "file_name", "empty", "thiing", "thing"])
    items = pd.read_csv(itempath, header = None, names=["item", "category", "thing", "thiing"])
    values = pd.read_csv(valuespath, header = None, names=["id", "calheaders_id", "item", "part", "value"])

    #edit the df to make it usefull
    headers.reset_index(inplace=True)
    headers.set_index("level_0", inplace=True)
    headers['date'] = pd.to_datetime(headers['date'], format='%d/%m/%Y')
    headers["date"] = headers["date"].astype("str")
    headers.columns = ["linac","date", "file_name", "empty", "thiing", "thing"]

    #merge the 3 files to get all the useful information
    tempdf = headers[["linac", "date", "file_name"]].merge(values[["calheaders_id", "item", "part", "value"]], left_on="level_0", right_on="calheaders_id", how="inner").drop(columns="calheaders_id")
    tempdf = tempdf.merge(items[["item", "category"]], on = "item", how="inner")

    return tempdf

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
@st.cache_data
def select_data_from_df(filter_on, df, col_n):
    df[col_n] = df[col_n].astype(str)
    return df[df[col_n].str.contains(fr'{str(filter_on)}', regex=True)]

"""
The db df common preperation for the status bar

input:
    df, dataframe: whole db df
    linac, string: linac code
return:
    this_df, datafram: whole db df edited
    dates, list: all the dates of that file
    df_changes, dataframe: contains per date all the changes between item and part
"""
def prep_db_to_status_bar(df, linac):
    # filter the df so that its just 1 file
    this_df = df[df["linac"] == linac].copy()
    this_df = this_df[this_df["part"] != 45]
    this_df = this_df[this_df["file_name"] == "Low X-ray energy - 6 MV"]

    #all dates
    dates = sorted(this_df["date"].unique().tolist())
    #determine last changed date
    changes_per_file = make_changes_per_file(this_df)
    #choose this certain file
    df_changes = changes_per_file["Low X-ray energy - 6 MV"]

    return this_df, dates, df_changes

"""
Create the linac status bar

input:
    df, dataframe: whole db df
    linac, string: linac code
return:
    output, dataframe: the values to be shown 
"""
@st.cache_data
def create_linac_status(df, linac):
    #do the df prep
    this_df, dates, df_changes = prep_db_to_status_bar(df, linac)

    #items of interest
    df_changes= df_changes[df_changes["item"].isin([381])]
    # make a list of the last changed date for all items
    last_changed_dates = df_changes[dates[1:]].apply(
    lambda row: row[(row != 0)].last_valid_index(), axis=1
    ).tolist()

    #addition of colour coding red if longer than 30 days
    last_changed = []
    for date in last_changed_dates:
        days = (pd.Timestamp.now().normalize() - pd.to_datetime(date)).days
        if days > 30:
            last_changed.append("🔴" + date)
        else:
            last_changed.append("🟢" + str(date))
    # to show
    output = pd.DataFrame({
            "Gun aim I": this_df[this_df["item"] == 381]["value"].values[0]
        }, index=["Value"]
    ).T
    output["Last changed"] = last_changed
    return output

"""
Make a list of all the changed part en items in the most recent change

input:
    df, dataframe: whole db df
    linac, string: linac code
return:
    output, list: the part and items to be shown 
"""
@st.cache_data
def make_last_changed_values(df, linac):
    #do the df prep
    this_df, dates, df_changes = prep_db_to_status_bar(df, linac)

    #all the last changed items
    recent_changed = df_changes[[dates[-1], "category"]]
    items_parts = recent_changed[recent_changed[dates[-1]] != 0]["category"].unique().tolist()
    return items_parts

"""
make the total amount of changes figure

input:
    df, dataframe: object that will be filterd
return:
    plotly line object
"""
def total_amount_fig(df):
    selected_df = select_data_from_df(st.session_state.linac_total_fig, df , col_n="linac")
    selected_df = select_data_from_df(st.session_state.type_file_total_fig, selected_df, col_n="file_name")

    changes_per_file = make_changes_per_file(selected_df)
    plotting_per_file = make_total_amount_to_plot(changes_per_file, st.session_state.see_part_45)
    st.session_state["changes_per_file"] = changes_per_file
    st.session_state["types_file_total_amount"] = sorted(list(set(plotting_per_file["File name"])))
    fig_total = px.line(plotting_per_file, x='Date', y='Amount', color='File name', markers=True, hover_data="First entry", title= "Total amount of differences per file for the selected linac")
    return fig_total


def filter_out_p45(df):
     #to not see part 45
    if st.session_state.see_part_45 == False:
        df = df[df["part"] != 45]
    return df


"""
make the object for the total mount plot

input:
    changes_per_file, dict: contains filenames, df_changes: df with the changes from last date 
    see_part_45, boolean: the output of a checkbox
return:
    plotting_per_file, dataframe for plotting the total amount plot
"""
@st.cache_data 
def make_total_amount_to_plot(changes_per_file, see_part_45):
    #get the sum of the amount of changes
    plotting_per_file = pd.DataFrame()
    for file_name, df_changes in changes_per_file.items():
            # get how many times there was a diffence between col
            date_columns = df_changes.columns.difference(['date', 'linac', 'file_name', 'item', 'part', 'category', df_changes.columns[-1]])

            df_changes = filter_out_p45(df_changes)

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
    df_pivot = df.pivot(index=["linac", "file_name", "item", "part", "category", "Resolution"], columns="date", values="value")
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
        df_changed = pd.concat([df_pivot[df_pivot["file_name"] == file_name].loc[:,["linac", "item", "part", "category"]].reset_index(), 
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
### complex???
@st.cache_data 
def grouped_plot(changes_per_file, groups, file_name):
    #get the correct df_changes
    df = changes_per_file[file_name]

    df = filter_out_p45(df)


    #get amount of changes per date
    date_columns = df.columns.difference(['date', 'linac', 'file_name', 'item', 'part', df.columns[-1]])
    #add the correct group after the correct item
    grouped = pd.merge(df, groups[["item", "part","groep"]], on=["item", "part"], how="left")
    
    #make the df for plotting of this data
    plotting_per_group = pd.DataFrame()
    for group in set(grouped["groep"]):
        df_changes = grouped[grouped["groep"] == group]
        #get de sum of amount of changes
        plotting = list((df_changes[date_columns] != 0).sum()[:-1]) 

        #get the changed items
        df_for_items = df_changes[df_changes[date_columns] != 0]

        
        try:
            for_items = df_for_items[date_columns].copy()
            # replace the amount with the category
            for col in date_columns:
                for_items[col] = df_for_items.apply(
                    lambda row: row["category"] if pd.notna(row[col]) else None,
                    axis=1
                )
            # get an list per date which items been changed
            items_plot = [for_items[col].dropna().unique().tolist() for col in date_columns]
        except:
            items_plot = [""] * len(date_columns) 
    
        first_date = re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-2])[0]

        data = {
            "Amount" : plotting,
            "Date": list(date_columns[:-1]),    
            "File name": [file_name] * len(plotting),
            "First entry": [first_date] * len(plotting),
            "Group": [group] * len(plotting),
            "item": items_plot[:-1]
        }

        plotting_per = pd.DataFrame(data) 
        plotting_per_group = pd.concat([plotting_per_group, plotting_per])
    fig = make_multiple_grouped_plot(plotting_per_group, split_on="Group", plot_on="Amount", hover_data = "item")
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
def make_multiple_grouped_plot(plotting, split_on, plot_on, hover_data):
    # get nessecary
    groups = list(set(plotting[split_on]))
    groups = [x for x in groups if str(x) != 'nan']

    if len(groups) == 0:
        return go.Figure()  # empyt figure

    # create fig
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, subplot_titles=groups)

    for i, group in enumerate(groups):
        df = plotting[plotting[split_on] == group]
        #the hoverdata
        custom = df[[hover_data]].values

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[plot_on],
                mode="lines+markers",
                name=group,
                showlegend=False,
                customdata=custom,
                hovertemplate=
                    "Datum: %{x}<br>" +
                    f"{plot_on}:" "%{y}<br>" +
                    f"{hover_data}:" "%{customdata[0]}<extra></extra>"
            ),
            row=i + 1, col=1
        )

    fig.update_layout(height=300 * len(groups), title_text="Meerdere Groepen Plot")
    return fig

    

"""
to make the bulk of the data smaller for compare df

input:
    df, dataframe
return:
    selected dataframe 
"""
@st.cache_data 
def select_linac(df, linac):
    return select_data_from_df(linac, df, col_n="linac")

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
    linac, string: the ouput of a selectbox
return:
    dataframe 
"""
def make_compare_df(df, linac):
    # grab only selected data
    newdf = select_linac(df, linac)
    newdf = filter_out_p45(newdf)

    # combine the 2 dates
    filterddf = pd.concat(
        [select_data_from_df(st.session_state.old_date_linac_compare_df1, newdf, col_n="date"), 
        select_data_from_df(st.session_state.new_date_linac_compare_df1, newdf, col_n="date")]
    )

    # calculate the difference
    changes_per_file = make_changes_per_file(filterddf)

    
    #to remove the item & part if has 0 difference
    if st.session_state.none_changes_compare_linacs == False:
        changes_per_file = {filename: df_changes[(df_changes.iloc[:,5:-1] != 0).any(axis=1)] for filename, df_changes in changes_per_file.items()}
    
    #make it to a df to show
    total_df = pd.DataFrame()
    for filename, df_changes in changes_per_file.items():

        if df_changes.columns[-3] != "part":

            old_data = df_changes[df_changes.columns[-1]]
            dif = df_changes[df_changes.columns[-2]]

            data = {
                    "File name": [filename] * len(df_changes),
                    "Item & part": "I:"+ df_changes["item"].astype(str) + " P:"+ df_changes["part"].astype(str),
                    "Difference": dif.tolist(),
                    "category": df_changes["category"].tolist(),
                    "Old data": old_data.tolist(),
                    "New data": list(old_data + dif)
                }

            temp_df = pd.DataFrame(data) 
            total_df = pd.concat([total_df, temp_df])

    
    return total_df



"""
To check if its nessecary to make of the one plot multiple plots if the y-axis has a wide spread.

input: 
    df, dataframe: that will be plotted
return:
    boolean, if a multiplot is nessecary
"""
def requires_multiple_plots_by_category(df):
    maxus = []
    #get the type of different kind is present
    items = list(set(df["Category"]))
    for item in items:
        maxus.append(int(df[df["Category"] == item]["Value"].max()))
    # determine the differences between the lowest and highest value of a kind
    s_maxus = sorted(maxus)
    if s_maxus[0] - s_maxus[-1] > 2000 or s_maxus[0] - s_maxus[-1] < -100:
        return True
    return False

"""
Detecting for outliers in the given data using SMA

input:
    group, dataframe: contains the data of a single group
output:
    group, dataframe: with the SMA values added

"""
def detect_outliers(group, sma_window=3, std_window=6, k=2):
    group = group.copy()
    group['SMA'] = group['Value'].rolling(window=sma_window).mean()
    group['residual'] = group['Value'] - group['SMA']
    group['rolling_std'] = group['residual'].rolling(window=std_window).std()
    group['is_outlier'] = np.abs(group['residual']) > (k * group['rolling_std'])
    return group

def make_ratio_BF_BC(df):
    # make the ratio
    ratio = df[df["Category"] == "Bending F (305)"].copy()
    ratio["ratio"] = df[df["Category"] == "Bending F (305)"]["Value"].values / df[df["Category"] == "Bending C (304)"]["Value"].values
    # difference in the ratio
    ratio_to_plot = ratio.iloc[1:]
    ratio_to_plot["ratio diff"] = ratio["ratio"].iloc[1:].values - ratio["ratio"].iloc[:-1].values

    return ratio_to_plot[ratio_to_plot["ratio diff"] != 0 ]

"""
make the plot to compare multiple part and items

input:
    changes_per_file, dict: filename: df with the changes per item
    ips, list: all the choosen catgories
    newdf, dataframe: the data that needs to be plotted
return:
    plotly line object
"""
### complex??
@st.cache_data 
def make_diff_plot_comparedf(changes_per_file, ips, newdf):
    #make the df to show
    total_df = pd.DataFrame()
    for filename, df_changes in changes_per_file.items():
        if df_changes.columns[-2] != "part":
            #add of the resolution
            df_changes = df_changes.merge(newdf[["Resolution", "item"]], on="item", how="left")
            dates = list(df_changes.columns)[5:-2]
            dates.append(re.findall(r"\d{4}-\d{2}-\d{2}", df_changes.columns[-2])[0])
            dates = sorted(dates)

            for ip in ips:
                ip_df = df_changes[df_changes["category"] == ip]
                #make the amount values again from the differences         
                first_entry = ip_df.iloc[:,-2].values[0]
                difference = ip_df[dates[1:]].values[0]
                total_list = list(accumulate([first_entry] + difference.tolist()))

                data = {
                        "Linac": [df_changes["linac"][1]] * len(dates),
                        "Category": [ip] * len(dates),
                        "Date": dates,
                        "Value": total_list,
                        "Part": [ip_df["part"].values[0]] * len(dates),
                        "Unit": [str(ip_df["Resolution"].values[0])] * len(dates) 
                    }
                temp_df = pd.DataFrame(data) 
                total_df = pd.concat([total_df, temp_df])
    #determine ouliers using SMA
    df_outlier_scanned = (
        total_df
        .groupby("Category", group_keys=False)
        .apply(detect_outliers)
    )
    # Filter outliers
    outliers = df_outlier_scanned[df_outlier_scanned["is_outlier"]]
    
    # to check if both the items are spread more than 100 apart
    if requires_multiple_plots_by_category(total_df):
        # make multiple line plots
        fig = make_multiple_grouped_plot(total_df, split_on="Category", plot_on="Value", hover_data="Unit")
        fig.update_layout(title_text="The value per category over the time", height=150*len(list(set(total_df["Category"]))))
    elif ["Bending F (305)", "Bending C (304)"] == ips:
        fig = px.line(total_df, x='Date', y='Value', color='Category', hover_data=["Unit", "Part"], markers=True, title= "The value per Item & part over the time")
        # find the differences of the ratios
        ratios_df = make_ratio_BF_BC(total_df)
        final_ratio = total_df[total_df["Category"].isin(["Bending F (305)", "Bending C (304)"])].copy()
        # do it for both of the lines
        final_ratio = ratios_df.merge(final_ratio[["Date", "Value"]], left_on="Date", right_on="Date", how="inner")
        fig.add_scatter(x=final_ratio["Date"],
                y=final_ratio["Value_y"],
                mode="markers",
                marker=dict(
                    color='Black',
                    size=5
                ),
               name='Ratio difference')
        
    else:
        #make one plot
        fig = px.line(total_df, x='Date', y='Value', color='Category', hover_data=["Unit", "Part"], markers=True, title= "The value per Item & part over the time")
        #add outliers as points
        fig.add_scatter(x=outliers["Date"],
                y=outliers["Value"],
                mode="markers",
                marker=dict(
                    color='black',
                    size=5
                ),
               name='Outliers')
    return fig

"""
Determine categories with change in them that will be shown on the page

input:
    df, dataframe: the data selected
    filetype, str: output of the chosen filetype
return:
    to_plot, changes_per_file
    ips, list: possible categories that have change
    newdf, dataframe: only has the chosen linac and filetype
"""
@st.cache_data 
def det_interesting_cat(df, filetype):
    newdf = select_data_from_df(filetype, df, "file_name")
    newdf = select_data_from_df(df=newdf, filter_on=st.session_state.linac_total_fig, col_n="linac")
    newdf = filter_out_p45(newdf)

    # make the differences per file
    changes_per_file = make_changes_per_file(newdf)
    to_plot = make_changes_per_file(newdf)

    #to remove the item & part if has 0 difference
    if st.session_state.none_changes_compare_linacs == False:
        changes_per_file = {filename: df_changes[(df_changes.iloc[:,5:-1] != 0).any(axis=1)] for filename, df_changes in changes_per_file.items()}

    ips = set(changes_per_file[filetype]["category"])
    return to_plot, ips, newdf

""" 
The filenames are numberd, translate it to energy levels

input, 
    df, dataframe, the used data
    translate, dataframe: the translate table
return,
    df, dataframe, the used data with translated file names
"""
@st.cache_data 
def translate_filename(df, translate):
    #get the translation for part 32
    trans = {filenam: translate[value] for value, filenam in df[df["item"] == 32][["value", "file_name"]].values if value <= 19 and value > 0}

    # apply the translation
    for filename, value in trans.items():
        df = df.replace(filename, value)
    return df

"""
Function to check if a value is numeric (either integer or float)
"""
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

"""
The value need to be transformed with a certain value, based usual names.

input: 
    dataframe, that needs to be translated
return:
    dataframe, that has been adjusted to the certain value for a item
"""
def translate_linacs(df):
    linacs = {2325: "linac1",
              2638: "linac2",
              2191: "linac3",
              2983: "linac4",
              7460: "linac5"}
    for linac in linacs:
        df = df.replace(linac, linacs[linac])
    return df

"""
The value need to be transformed with a certain value, based on the elekta manual.

input: 
    dataframe, that needs to be translated
return:
    dataframe, that has been adjusted to the certain value for a item
"""
def translate_category(df): 
    linac_items = st.session_state.linac_items
    tempdf = df[df["file_name"].str.contains("Mlc|Optics")]
    tempdf["category"]= tempdf["category"] + " (" + tempdf["item"].astype("str") + "," + tempdf["part"].astype("str") + ")"

    otherdf = df[~df["file_name"].str.contains("Mlc|Optics", na=False)]
    #translate the category
    otherdf = otherdf.merge(linac_items[["Category", "Item name"]], left_on="item", right_on="Item name", how="left").drop(columns=["Item name", "category"])
    #add item number
    otherdf["category"] = otherdf["Category"] + " (" + otherdf["item"].astype("str") + "," + otherdf["part"].astype("str") + ")"
    otherdf = otherdf.drop(columns="Category")
    return pd.concat([otherdf, tempdf])

"""
editing of the database data
"""
def edit_db_data(df):
    # to translate energy filename
    # 2 and 5 changed
    energy_dict = {
        0: "No energy set",
        1: "Low X-ray energy - 6 MV",
        2: "Third X-ray energy - 10 MV",
        3: "energy: Radiographic X-ray",
        4: "High X-ray energy - 25 MV",
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

    df = translate_linacs(df)

    df = translate_category(df)
    print(df)

    return df

# read data
df_db = get_db_data("data/calblocks_update_table_cal_headers.csv", 
                    "data/calblocks_update_table_cal_items.csv",
                    "data/calblocks_update_table_cal_values.csv")
# edit the data
linac_items = get_transl_data("data/translate_tbl.csv")
linac_items["Item description"] = linac_items["Item name"].str.replace(r'^i\d+\s*-\s*', '', regex=True)
linac_items["Item name"] =  linac_items["Item name"].str.extract(r'i(\d+)').astype("float")
st.session_state["linac_items"] = linac_items
df_db = edit_db_data(df_db)

#load grouping
grouped_partly = get_grouped_data("data/groups_partly.tsv")


# layout
layout = st.container(border=True)
layout.title("Linac logfiles comparing")

#linac status part
linac_status = st.container(border=True)
linac_status.write("Current linac status at certain items")
linac_status_1, linac_status_2 = linac_status.columns(2)
l1 = linac_status_1.container()
l2 = linac_status_1.container()
l3 = linac_status_1.container()
l4 = linac_status_2.container()
l5 = linac_status_2.container()

l1.write("*Linac 1*")
l1.write(create_linac_status(df_db, "linac1"))
l1.write("**Last changed values:**\n" + "\n".join(f"- {line}" for line in make_last_changed_values(df_db, "linac1")))
l2.write("*Linac 2*")
l2.write(create_linac_status(df_db, "linac2"))
l2.write("**Last changed values:**\n" + "\n".join(f"- {line}" for line in make_last_changed_values(df_db, "linac2")))
l3.write("*Linac 3*")
l3.write(create_linac_status(df_db, "linac3"))
l3.write("**Last changed values:**\n" + "\n".join(f"- {line}" for line in make_last_changed_values(df_db, "linac3")))
l4.write("*Linac 4*")
l4.write(create_linac_status(df_db, "linac4"))
l4.write("**Last changed values:**\n" + "\n".join(f"- {line}" for line in make_last_changed_values(df_db, "linac4")))
l5.write("*Linac 5*")
l5.write(create_linac_status(df_db, "linac5"))
try:
    l5.write("**Last changed values:**\n" + "\n".join(f"- {line}" for line in make_last_changed_values(df_db, "linac5")))
except:
    l5.write("Misses a item")

#the first total amount plot
amount_plot = st.container(border= True)
a1, a2, a3 = amount_plot.columns(3)
a1.checkbox("See part 45", key="see_part_45")
a2.selectbox(label="The type of file you want to see", options=["energy", "Be63", "Optics", "Mlc"], key="type_file_total_fig")
a3.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_total_fig")
amount_plot.plotly_chart(total_amount_fig(df_db))
amount_plot.selectbox(label="The type of file you want to see", options=st.session_state.types_file_total_amount, key="type_file_grouped_fig")
#the grouped amount plot
grouped_amount_plot = amount_plot.expander(label="See the changes per group")
grouped_amount_plot.write("The groups are based on the system drawings, see [here](https://isala.sharepoint.com/sites/AFDRadiotherapie/RT%20afdelingsschijf/Forms/AllItems.aspx?id=%2Fsites%2FAFDRadiotherapie%2FRT%20afdelingsschijf%2FKlinische%20Fysica%2FOpenbaar%2FProducthandleidingen%2FElekta%20Versneller%2FEVO%2FLinac%20%2D%20System%20Diagrams%20%28from%20Linac%20152972%29%20%2D%201543441%5F04%2Epdf&parent=%2Fsites%2FAFDRadiotherapie%2FRT%20afdelingsschijf%2FKlinische%20Fysica%2FOpenbaar%2FProducthandleidingen%2FElekta%20Versneller%2FEVO).")
grouped_amount_plot.plotly_chart(grouped_plot(st.session_state.changes_per_file, grouped_partly, st.session_state.type_file_grouped_fig))

# set up the elements for the second: difference plot
diff_plot = st.container(border=True)

# set up the elements for the third: compare linacs df
compare_linacs = st.container(border=True)
cl1, cl2, cl3 = compare_linacs.columns(3)
cl1.checkbox("See the Part & items that had no changes", key="none_changes_compare_linacs")

# set up the working elements of the second
btn1, btn2 = diff_plot.columns(2)
btn1.selectbox(label='File type', options=st.session_state.types_file_total_amount, key="file_type_diff_plot")
btn2options = list(df_db[df_db["file_name"] == st.session_state.file_type_diff_plot]["category"].unique())
changes_per_file, ips_options, newdf = det_interesting_cat(df_db, st.session_state.file_type_diff_plot)
diff_plot.write(ips_options)

# The mulitselect needs a security to remeber all
if "i_and_p" not in st.session_state:
    st.session_state.i_and_p = []

def update_selection():
    st.session_state.i_and_p = st.session_state.temp_i_and_p

# plotting of the difference plot
try:
    btn2.multiselect(label="Part & item combination", options=btn2options, default=st.session_state.i_and_p, key="temp_i_and_p", on_change=update_selection)
    diff_plot.plotly_chart(make_diff_plot_comparedf(changes_per_file, st.session_state.i_and_p, newdf))
except Exception as e:
    diff_plot.write("Choose a Part & item you want to see")

#set up de working elemtens of the third
multiple1, multiple2  = compare_linacs.columns(2)
compare_dfs = compare_linacs.expander(label= "see the dataframes with the selected data")
df1, df2  = compare_dfs.columns(2)
compare_plot = compare_linacs.expander(label= "see the plot with the difference of the selected data")

#make the compare linacs dataframes
multiple1.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df1")
multiple2.selectbox(label="The Linac you want to see", options=sorted(list(set(df_db["linac"]))), key="linac_linac_compare_df2")
cl2.selectbox(label="The date you want to compare to", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"])), reverse=True), key="old_date_linac_compare_df1")
cl3.selectbox(label="The date you want to see the difference of", options=sorted(list(set(df_db[df_db["linac"] == st.session_state.linac_linac_compare_df1]["date"])), reverse=True)[1:], key="new_date_linac_compare_df1")
compare_df = df1.data_editor(make_compare_df(df_db, st.session_state.linac_linac_compare_df1), key="linac_compare_df")
compare_df2 = df2.data_editor(make_compare_df(df_db, st.session_state.linac_linac_compare_df2), key="linac_compare_df2")
