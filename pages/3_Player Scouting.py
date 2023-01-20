import pandas as pd
import streamlit as st

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Veri Okuma / Player Database
@st.cache()
def read_data():
    df = pd.read_excel(r"FM_2023_final.xlsx")
    df.set_index("Name", inplace=True)
    return df

df = read_data()
df.head()

# Unique Nationality
lst= df["Nationality"].unique().tolist()
liste1=[]
for i in lst:
    liste1.append(i.split(","))
l1 = []
def flatten(n):
    for i in n :
        if isinstance(i,list):
            flatten(i)
        else:
            l1.append(i)
            
flatten(liste1)

Natlist=pd.DataFrame(l1)[0].unique().tolist()

# Similar Players

def calculate_corr(dataframe, playername, number=3000):
    # Abilities
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
    ability = Techn + Attack + Power + Speed + Defence + Mentality + GoalK
    # Corr func
    dataframe = dataframe[ability]
    corr_table = dataframe.T.corr()[[playername]].sort_values(by=playername,ascending=False)[1:number+1]
    corr_table.reset_index(inplace=True)
    corr_table.rename(columns={playername:"Correlation"}, inplace=True)
    return corr_table

def final_table(correlationtable, firstdataframe, position, nationality,maxage, minheight, maxweight):
    df2=firstdataframe.reset_index()
    final_cordf = pd.merge(similar_players, df2, on="Name")
    final_cordf = final_cordf[["Name", "Correlation", "Position.1", "Last_Player_Value", "Age", "Nationality", "Height", "Weight"]]
    final_cordf = final_cordf[final_cordf["Position.1"]==position]
    final_cordf = final_cordf[final_cordf["Age"] <= maxage]
    final_cordf = final_cordf[final_cordf["Nationality"].str.contains(nationality)]
    final_cordf = final_cordf[final_cordf["Height"] >= minheight]
    final_cordf = final_cordf[final_cordf["Weight"] <= maxweight]
    return final_cordf

playernames = list(df.index.unique())
targetplayer = st.selectbox("Please choose a target player to find similars", playernames)
targetpos = st.selectbox("Please choose a position to filter similar players", list(df["Position.1"].unique()))
targetcountry = st.selectbox("Please choose a nationality to filter similar players", Natlist)
filterage = st.slider("Please choose maximum age to filter similar players", min_value=15,max_value=41)
filterminheight = st.slider("Please choose min height to filter similar players", min_value=159,max_value=204)
filtermaxweight = st.slider("Please choose max weight to filter similar players", min_value=55,max_value=101)


if st.button("Find similar players"):
    similar_players=calculate_corr(df, targetplayer)
    finaldf = final_table(similar_players, df, targetpos, targetcountry, filterage, filterminheight, filtermaxweight)
    st.write(finaldf)