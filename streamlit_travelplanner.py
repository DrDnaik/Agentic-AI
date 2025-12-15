import streamlit as st
from travel_planner import run_travel_planning
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

if 'planning_complete' not in st.session_state:
    st.session_state.planning_complete = False

if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown('<center> <h1> ✈️ AI Travel Planner </h1></center> ', unsafe_allow_html=True)
st.markdown("<center> <h4> Your Intelligent Multi-Agent Travel Assistant</h4> <center>", unsafe_allow_html=True)

with st.sidebar:
    st.header(' Trip Requirements : ')
    st.markdown('---')

    destination = st.text_input(
        "Destination",
        placeholder="eg : Paris, Tokya, Bali etc.",
        help="Enter the city you wanna visit"
    )

    days = st.number_input(
        "Number of days",
        min_value=1,
        max_value=30,
        value=5,
        help="How many days will you be travelling ? "
    )

    travel_month = st.selectbox(
        "Travel Month",
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "Sepetember", "October", "November", "December"],
        index=datetime.now().month - 1,
        help="Which month will you be travelling ? "
    )

    budget = st.selectbox(
        "Budget Level",
        ["Low", "Medium", "High", "Luxury"],
        index=1,
        help="Select your budget preference! "
    )

    interests = st.multiselect(
        "Interests",
        ["Nature", "Adventure", "Food", "Shopping", "Beach", "Nightlife", "Hiking", "Culture", "Photography"],
        default=["Adventure", "Food"],
        help="Select your interests! "
    )

    group_type = st.radio(
        "Travel Group",
        ["Solo", "Couple", "Family", "Friends", "Work"],
        help="Select your travel group"
    )

    nationality = st.text_input(
        "Nationality",
        value="Indian",
        help="Your nationality"
    )

    st.markdown('---')

    plan_button = st.button("Create my Travel Plan", use_container_width=True)

if plan_button:

    if not destination:
        st.error("Please enter a destination to continue!")
    else:

        st.session_state.planning_complete = False

        user_requirements = {
            'destination': destination,
            'days': days,
            'budget': budget,
            'interests': ", ".join(interests),
            'travel_month': travel_month,
            'group_type': group_type,
            'nationality': nationality
        }

        st.success("✅ Requirements collected! Starting AI travel Planning ....")

        with st.expander("Your trip Summary", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Destination", destination)
                st.metric("Duration", f"{days} days")

            with col2:
                st.metric("Budget", budget)
                st.metric("Travel Month", travel_month)

            with col3:
                st.metric("Group Type", group_type)
                st.metric("interests", ", ".join(interests))

        with st.spinner("AI agents are crafting your travel plan ...This may take a few minutes! "):
            results = run_travel_planning(user_requirements)
            st.session_state.planning_complete = True
            st.session_state.results = results

if st.session_state.planning_complete and st.session_state.results:
    travel_results = st.session_state.results

    st.markdown('---')

    tabs = st.tabs([
        "Final Intinerary", "Weather", "Transport", "Accomodation", "Activities",
        "Budget", "Safety Tips", "Documents"
    ])

    with tabs[0]:
        st.markdown('### Complete Travel Itinerary')
        st.info(travel_results.get('final_itinerary', 'No itinerary generated'))

    with tabs[1]:
        st.markdown('### Weather and Season Advisor')
        st.info(travel_results.get('weather', 'No weather info generated'))

    with tabs[2]:
        st.markdown('### Transport Plan')
        st.info(travel_results.get('transport', 'No transport info generated'))

    with tabs[3]:
        st.markdown('### Accomodation Recommendations')
        st.info(travel_results.get('accomodation', 'No accomodation info generated'))

    with tabs[4]:
        st.markdown('### Activities & Attractions')
        st.info(travel_results.get('activities', 'No activities info generated'))

    with tabs[5]:
        st.markdown('### Budget Breakdown')
        st.info(travel_results.get('budget', 'No budget info generated'))

    with tabs[6]:
        st.markdown('### Safety and Local Tips')
        st.info(travel_results.get('safety', 'No safety info generated'))

    with tabs[7]:
        st.markdown('### Travel Documents and Visa')
        st.info(travel_results.get('documents', 'No documents info generated'))

    st.markdown('---')


















