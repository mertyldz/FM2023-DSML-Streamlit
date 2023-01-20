import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Scout")
st.title("FM23 Talent Scout Work - Scout")

# Player Database
@st.cache()
def read_data():
    df = pd.read_excel(r"FM_2023_final.xlsx")
    df.set_index("Name", inplace=True)
    df.drop(["Unnamed: 0", "UID", "Unnamed: 102", "Unnamed: 103"], axis=1, inplace=True)
    return df
df = read_data()
st.header("Player Database")
st.dataframe(df)

player_names = list(df.index.unique())

# Player Stats
st.header("Player Stats")
st.subheader("SWOT Analysis")

@st.cache()
def createSwot(dataframe, playername):
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
    Attributess = Techn + Attack + Power + Speed + Defence + Mentality + GoalK
    if playername in dataframe[dataframe["Position.1"] == "Goalkeeper"].index:
       dataframe1 = dataframe[dataframe.index == playername][Attributess].T.sort_values(by=playername, ascending=False).head(5)
       dataframe2 = dataframe[dataframe.index == playername][Attributess].T.sort_values(by=playername, ascending=False).tail(5)
       swot=pd.concat([dataframe1, dataframe2])
       return swot
    else:
        Attributess = [col for col in Attributess if col not in GoalK]
        dataframe1 = dataframe[dataframe.index == playername][Attributess].T.sort_values(by=playername, ascending=False).head(5)
        dataframe2 = dataframe[dataframe.index == playername][Attributess].T.sort_values(by=playername,ascending=False).tail(5)
        swot = pd.concat([dataframe1, dataframe2])
        return swot
    

swotplayer = st.selectbox("Give a player name to see strengths and weaknesses", player_names)

if st.button("Show SWOT table of player"):
    st.text(f"Attributes of {swotplayer}")
    st.write(df[df.index == swotplayer])
    st.text(f"Strongest and weakest 5 attributes of {swotplayer}")
    st.write(createSwot(df, swotplayer))

# Player visuzalization
st.subheader("Player Visuzalization")
@st.cache()
def create_dendrogram(dataframe, playername):
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
    Attributess = Techn + Attack + Defence + Power + Speed + Mentality + GoalK
    dataframe = dataframe[dataframe.index == playername][Attributess].T
    link=linkage(dataframe, "ward")
    plt.figure(figsize=(16, 9))
    dendrogram(link, leaf_font_size=10, orientation="right", labels=dataframe.index)
    plt.ylabel(f"Attributes of {playername}")
    plt.xlabel("Distances")
    plt.title(f"Attribute Dendrogram of {playername}")
    plt.show(block=True)

@st.cache()
def create_attributelist():
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
    attributess = Techn + Attack + Defence + Power + Speed + Mentality + GoalK
    return attributess

if st.button("Player visualization"):
    attributess = create_attributelist()
    st.bar_chart(df[df.index==swotplayer][attributess].T)
    st.pyplot(create_dendrogram(df, swotplayer))

# Player Clustering
st.header("Player Clustering")
pos = st.selectbox("Choose Player Position for Clustering",list(df["Position.1"].unique()))

def elbowplotter(dataframe, numeric_columns, position):
    dataframe = dataframe[dataframe["Position.1"] == position]
    dataframe = dataframe[numeric_columns]
    scaler = RobustScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns = dataframe.columns, index=dataframe.index)
    kmeans = KMeans(random_state=1601).fit(dataframe)
    visualizer = KElbowVisualizer(kmeans, k=(1,15)).fit(dataframe)
    st.pyplot()

@st.cache()
def create_kmeans(dataframe, numeric_columns, position):
    dataframe = dataframe[dataframe["Position.1"] == position]
    dataframe = dataframe[numeric_columns]
    scaler = RobustScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns = dataframe.columns, index=dataframe.index)
    kmeans = KMeans(random_state=1601).fit(dataframe)
    visualizer = KElbowVisualizer(kmeans, k=(1,15)).fit(dataframe)
    kmeans_final = KMeans(random_state=1601, n_clusters=visualizer.elbow_value_).fit(dataframe)
    dataframe = pd.DataFrame(scaler.inverse_transform(dataframe), columns=dataframe.columns, index=dataframe.index)
    dataframe["Clusters"] = kmeans_final.labels_
    dataframe["Clusters"] = dataframe["Clusters"]+1
    return dataframe

if st.button("Cluster by position"):
    attributess = create_attributelist()
    elbowplotter(df, attributess, pos)
    clusters=create_kmeans(df, attributess, pos)
    st.write(clusters)
    st.write(clusters.groupby("Clusters").describe())

# PCA
st.header("PCA")

pos1 = st.selectbox("Choose Player Position for PCA",list(df["Position.1"].unique()))

@st.cache()
def create_PCA(dataframe, position):
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
    if position == "Goalkeeper":
        Attributess = Techn + Attack + Defence + Power + Speed  + Mentality + GoalK
    elif position == "Forward":
        Attributess = Techn + Attack + Power + Speed + Mentality
    elif position == "Defender":
        Attributess = Techn + Power + Speed + Defence + Mentality
    else:
        Attributess = Techn + Attack + Power + Speed + Defence + Mentality
    dataframe = dataframe[dataframe["Position.1"]==position]
    dataframe = dataframe[Attributess]
    pca = PCA(n_components=10)
    pca_fit = pd.DataFrame(pca.fit_transform(dataframe), index=dataframe.index)
    # KARA VERMEK GEREK!!!
    pca_fit=pca_fit.applymap(lambda x: abs(x))
    pca_fit["Score"] = pca_fit.apply(lambda x: x.mean(), axis=1)
    cols_to_drop = [col for col in pca_fit.columns if col not in ["Score"]]
    pca_fit.drop(cols_to_drop, axis=1, inplace=True)
    pcafit2 = pca_fit.copy()

    pca_fit = pca_fit.sort_values(by="Score", ascending=False).head(10)
    return pca_fit

@st.cache()
def club_PCA(dataframe, position):
    club = dataframe["Club"]
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
    if position == "Goalkeeper":
        Attributess = Techn + Attack + Defence + Power + Speed  + Mentality + GoalK
    elif position == "Forward":
        Attributess = Techn + Attack + Power + Speed + Mentality
    elif position == "Defender":
        Attributess = Techn + Power + Speed + Defence + Mentality
    else:
        Attributess = Techn + Attack + Power + Speed + Defence + Mentality
    dataframe = dataframe[dataframe["Position.1"]==position]
    dataframe = dataframe[Attributess]
    pca = PCA(n_components=10)
    pca_fit = pd.DataFrame(pca.fit_transform(dataframe), index=dataframe.index)
    # KARA VERMEK GEREK!!!
    pca_fit=pca_fit.applymap(lambda x: abs(x))
    pca_fit["Score"] = pca_fit.apply(lambda x: x.mean(), axis=1)
    cols_to_drop = [col for col in pca_fit.columns if col not in ["Score"]]
    pca_fit.drop(cols_to_drop, axis=1, inplace=True)
    pca_fit2 = pca_fit.copy()
    clubpca = pd.concat([pca_fit2, club], axis=1)
    clubpca = clubpca.groupby("Club").agg({"Score":"mean"}).sort_values(by="Score",ascending=False)
    return clubpca


if st.button("Create highest PCA score list"):
    st.write(create_PCA(df,pos1))
    st.write(club_PCA(df, pos1))

# Correlation
st.header("Correlation")
@st.cache()
def create_correlation(dataframe, league, position, xvar, yvar):
    dataframe = dataframe[(dataframe["Position.1"] == position) & (dataframe["Leagues"] == league)]
    dataframe = dataframe[[xvar, yvar]]
    pearson=dataframe.corr(method="pearson").iloc[0,1]
    spearman=dataframe.corr(method="spearman").iloc[0,1]
    kendall=dataframe.corr(method="kendall").iloc[0,1]
    metrics = pd.DataFrame({"Pearson":[pearson], "Spearman":[spearman], "Kendall":[kendall]})
    return metrics, dataframe
create_correlation(df, "LaLiga", "Forward", "Stamina", "Pace")
attributess = create_attributelist()
league=st.selectbox("League", list(df["Leagues"].unique()))
pos3=st.selectbox("Position", list(df["Position.1"].unique()))
var1=st.selectbox("First Attribute", attributess)
var2=st.selectbox("League", [col for col in attributess if col not in var1])


if st.button("Create correlation"):
    metrics, dataframe = create_correlation(df, league, pos3, var1, var2)
    st.write(metrics)
    st.write(dataframe)
    fig = px.scatter(df, var1, var2, color=df.index)
    st.write(fig)
