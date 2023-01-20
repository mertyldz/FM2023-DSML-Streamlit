import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from soccerplots.radar_chart import Radar
from scipy.spatial.distance import pdist, squareform
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title = "Players")

st.title("FM23 Talent Scout Work - Players")
st.subheader("Dataset")
# READ DATA
def read_data():
    df = pd.read_excel(r"FM_2023_final.xlsx")
    df.set_index("Name", inplace=True)
    df.drop(["Unnamed: 0", "UID", "Unnamed: 102", "Unnamed: 103"], inplace=True,axis=1)
    return df
df = read_data()
st.dataframe(df)
player_columns = ["Age", "Last_Player_Value", "Salary", "Nationality"]
player_names = list(df.index.unique())


# COMPARE PLAYERS
st.subheader("Player Comparison")

def player_compare(dataframe, player1, player2, columns):
    compared_df = dataframe.loc[(dataframe.index == player1) | (dataframe.index == player2), columns]
    return compared_df

name1 = st.selectbox("First Player Name", player_names)
name2 = st.selectbox("Second Player Name", [name for name in player_names if name not in name1])

if st.button("Compare players"):
    st.write(player_compare(df, name1, name2, player_columns))

# RADAR CHART
st.subheader("Radar Chart")
@st.cache()
def radar_graph(dataframe, plot=False):
    Techn = ['Crossing', 'Dribbling', 'First Touch', 'Corners', 'Free Kick Taking', 'Technique', 'Passing', 'Left Foot',
             'Right Foot']
    Attack = ['Finishing', 'Heading', 'Long Shots', 'Penalty Taking', 'Jumping Reach']
    Power = ['Strength', 'Natural Fitness']
    Speed = ['Acceleration', 'Agility', 'Balance', 'Pace', 'Stamina']
    Defence = ['Marking', 'Tackling', 'Aggressiion', 'Long Throws', 'Foul']
    Mentality = ['Emotional control', 'Sportsmanship', 'Resistant to stress', 'Professional', 'Bravery', 'Anticipation',
                 'Composure', 'Concentration', 'Decision', 'Determination', 'Flair', 'Leadership', 'Work Rate',
                 'Teamwork', 'Stability', 'Ambition', 'Argue', 'Loyal', 'Adaptation', 'Vision', 'Off The Ball']
    GoalK = ['Reflexes', 'Kicking', 'Handling', 'One On Ones', 'Command Of Area', 'Communication', 'Eccentricity',
             'Rushing Out', 'Punching', 'Throwing', 'Aerial Reach']

    dataframe["Techn"] = dataframe[Techn].apply(lambda x: x.mean(), axis=1)
    dataframe["Attack"] = dataframe[Attack].apply(lambda x: x.mean(), axis=1)
    dataframe["Power"] = dataframe[Power].apply(lambda x: x.mean(), axis=1)
    dataframe["Speed"] = dataframe[Speed].apply(lambda x: x.mean(), axis=1)
    dataframe["Mentality"] = dataframe[Mentality].apply(lambda x: x.mean(), axis=1)
    dataframe["GoalK"] = dataframe[GoalK].apply(lambda x: x.mean(), axis=1)
    grouped_attributes = ["Techn", "Attack", "Power", "Speed", "Mentality", "GoalK"]
    df_radar = dataframe[grouped_attributes]
    return df_radar

radar1 = st.selectbox("First Player Name for Radar Graph", player_names)
radar2 = st.selectbox("Second Player Name for Radar Graph", [name for name in player_names if name not in radar1])

def create_radarchart(dataframe, player1, player2):
    params = list(dataframe.columns)
    dfradar = dataframe.reset_index()
    ranges = []
    a_values = []
    b_values = []
    for x in params:
        a = min(dfradar[params][x])
        a = a - (a*.25)
        
        b = max(dfradar[params][x])
        b = b + (b*.25)
        ranges.append((a,b))
    for x in range(len(dfradar["Name"])):
        if dfradar["Name"][x] == player1:
            a_values = dfradar.iloc[x].values.tolist()
        if dfradar["Name"][x] == player2:
            b_values = dfradar.iloc[x].values.tolist()
    
    a_values = a_values[1:]
    b_values = b_values[1:]
    values = [a_values, b_values]
    title = dict(
    title_name = player1,
    title_color = "green",
    title_name_2 = player2,
    title_color_2 = "red",
    title_fontsize = 18 )
    radar = Radar()
    fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=["green", "red"], alphas=[.5,.3], title=title, compare=True)
    st.pyplot()

if st.button("Create radar table"):
    radar = radar_graph(df)
    fig = create_radarchart(radar,radar1, radar2)

# BAR CHART
st.subheader("Compare with bar graphs")

bar1 = st.selectbox("First Player Name for Bar Graph", player_names)
bar2 = st.selectbox("First Player Name for Bar Graph", [name for name in player_names if name not in bar1])

@st.cache()
def barplot(dataframe, plot=False):
    Techn = ['Crossing', 'Dribbling', 'First Touch', 'Corners', 'Free Kick Taking', 'Technique', 'Passing', 'Left Foot',
             'Right Foot']
    Attack = ['Finishing', 'Heading', 'Long Shots', 'Penalty Taking', 'Jumping Reach']
    Power = ['Strength', 'Natural Fitness']
    Speed = ['Acceleration', 'Agility', 'Balance', 'Pace', 'Stamina']
    Defence = ['Marking', 'Tackling', 'Aggressiion', 'Long Throws', 'Foul']
    Mentality = ['Emotional control', 'Sportsmanship', 'Resistant to stress', 'Professional', 'Bravery', 'Anticipation',
                 'Composure', 'Concentration', 'Decision', 'Determination', 'Flair', 'Leadership', 'Work Rate',
                 'Teamwork', 'Stability', 'Ambition', 'Argue', 'Loyal', 'Adaptation', 'Vision', 'Off The Ball']
    barplot_attributes = Techn + Attack + Power + Speed + Defence + Mentality
    barplot = dataframe[barplot_attributes]
    return barplot

if st.button("Create bar graphs"):
    barplot = barplot(df)
    # st.bar_chart(barplot[barplot.index == bar1].T)
    # st.bar_chart(barplot[barplot.index == bar2].T)
    st.bar_chart(barplot[(barplot.index == bar1) | (barplot.index == bar2)].T, )

# FIND MOST SIMILAR PLAYERS
st.subheader("Similarity")

goal_player = st.selectbox("Choose a Player Name to Find Similars", player_names)

metric=st.selectbox("Metrics", ["euclidean","canberra","minkowski","correlation","cityblock","jaccard"], index=0)
# metric=st.text_input(metrics) #text input by writing

@st.cache()
def similarity_df(dataframe):
    Techn = ['Crossing', 'Dribbling', 'First Touch', 'Corners', 'Free Kick Taking', 'Technique', 'Passing', 'Left Foot',
             'Right Foot']
    Attack = ['Finishing', 'Heading', 'Long Shots', 'Penalty Taking', 'Jumping Reach']
    Power = ['Strength', 'Natural Fitness']
    Speed = ['Acceleration', 'Agility', 'Balance', 'Pace', 'Stamina']
    Defence = ['Marking', 'Tackling', 'Aggressiion', 'Long Throws', 'Foul']
    Mentality = ['Emotional control', 'Sportsmanship', 'Resistant to stress', 'Professional', 'Bravery', 'Anticipation',
                 'Composure', 'Concentration', 'Decision', 'Determination', 'Flair', 'Leadership', 'Work Rate',
                 'Teamwork', 'Stability', 'Ambition', 'Argue', 'Loyal', 'Adaptation', 'Vision', 'Off The Ball']
    GoalK = ['Reflexes', 'Kicking', 'Handling', 'One On Ones', 'Command Of Area', 'Communication', 'Eccentricity',
             'Rushing Out', 'Punching', 'Throwing', 'Aerial Reach']
    position = ["DL", "DC", "DR", "WBL", "WBR", "DM", "ML", "MC", "MR", "AML", "AMC", "AMR", "ST", "GK"]
    similarity_cols = Techn + Attack + Power + Speed + Defence + Mentality + GoalK + position
    similarity = dataframe[similarity_cols]
    return similarity

@st.cache()
def create_distance(dataframe, metric, plot=False):
    from scipy.spatial.distance import pdist, squareform
    distance = pdist(dataframe, metric=metric)
    matrix = squareform(distance)
    distance_df = pd.DataFrame(matrix, columns=dataframe.index, index=dataframe.index)
    return distance_df

@st.cache()
def most_similar(dataframe, number=5):
    euc_list={}
    for i in list(dataframe.columns):
        euc_list.update({i:list(dataframe[i].sort_values(ascending=True)[1:number+1].index)})
    ec=pd.DataFrame(euc_list)
    ec = ec[goal_player]
    return ec

if st.button("Most similar players"):
    simil = similarity_df(df)
    similaritydf = create_distance(simil, metric)
    mostsim15 = most_similar(similaritydf, 15)
    st.dataframe(mostsim15)
