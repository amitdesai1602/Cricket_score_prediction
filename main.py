import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
import streamlit as st
import  streamlit_option_menu
from streamlit_option_menu import option_menu
import sklearn
from PIL import Image


pipe = pickle.load(open('newmodel.pkl', 'rb'))


teams = ["Australia",
         'India',
         'Bangladesh',
         'New Zealand',
         'South Africa',
         'England󠁧󠁢󠁥󠁮',
         'West Indies',
         'Afghanistan',
         'Pakistan',
         'Sri Lanka']

cities = ['Colombo',
          'Mirpur',
          'Johannesburg',
          'Dubai',
          'Auckland',
          'Cape Town',
          'London',
          'Pallekele',
          'Barbados',
          'Sydney',
          'Melbourne',
          'Durban',
          'St Lucia',
          'Wellington',
          'Lauderhill',
          'Hamilton',
          'Centurion',
          'Manchester',
          'Abu Dhabi',
          'Mumbai',
          'Nottingham',
          'Southampton',
          'Mount Maunganui',
          'Chittagong',
          'Kolkata',
          'Lahore',
          'Delhi',
          'Nagpur',
          'Chandigarh',
          'Adelaide',
          'Bangalore',
          'St Kitts',
          'Cardiff',
          'Christchurch',
          'Trinidad']


st.set_page_config(
    page_title="Crics-11",
    page_icon="🏏"
)

with st.sidebar:
    select = option_menu("Menu",["Home","Project","Help"],
                         icons=['house','book','github'],
                         menu_icon="cast",default_index=0,
                         styles={"container": {"padding": "5!important", "background-color": "black"},
                                 "icon": {"color": "white", "font-size": "25px"},
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                              "--hover-color": "#262730"},
                                 "nav-link-selected": {"background-color": "red"},
                                 }
                         )

if select == "Home":
    ti1,ti2,ti3 = st.columns([0.3,1,1])
    ti2.title("Crics 11")
    image1 = Image.open('crics11.jpg', 'r')
    new_image = image1.resize((300,290))
    clo1,clo2,clo3 = st.columns([0.2,5,0.2])

    clo2.image(new_image)
    st.markdown("""
    <style>
    .big-font {
         font-size:28px !important;
    }
    </style>
    """, unsafe_allow_html=True
    )
    st.markdown('<p class="big-font">About Project</p>', unsafe_allow_html=True)
    st.write("   ")
    st.write("**In this dataset contain ball-by-ball data of Men's T20 International matches till 2020**")
    st.write("**This project will help to predict 1st inning score based on current match situation**")
    st.write("   ")
    st.write("   ")
    st.write("   ")



if select == "Project":
    st.title("Cricket Score Prediction")
    st.markdown("""
    <style>
    .big-font {
         font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True
    )
    st.markdown('<p class="big-font">Khelo Dimag Se🏏</p>', unsafe_allow_html=True)
    image = Image.open('crick1.jpg', 'r')

    st.image(image)

    col1, col2 = st.columns(2)

    with col1:
        batting_team = teams.index(st.selectbox('Select batting team', sorted(teams)))
    with col2:
        bowling_team = teams.index(st.selectbox('Select bowling team', sorted(teams)))

    city = cities.index(st.selectbox('Select city', sorted(cities)))

    col3, col4, col5 = st.columns(3)

    with col3:
        current_score = st.number_input('Current Score')
    with col4:
        overs = st.number_input('Overs done(works for over>5)')
    with col5:
        wickets = st.number_input('Wickets out')

    last_five = st.number_input('Runs scored in last 5 overs')


    if st.button('Predict Score'):
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame(
            {'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
             'last_five': [last_five]})
        result = pipe.predict(np.array([[batting_team, bowling_team, city, current_score, overs, wickets, crr, last_five]]))[0]
        st.write(result)
        st.header("Predicted Score - " + str(int(result)))




if select == "Help":
    st.title("Help")

    st.write("    ")
    st.write("    ")


    st.write("**Github Link: https://github.com/amitdesai1602/score-predtication-cricket**")
    st.write("    ")




