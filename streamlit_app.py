import streamlit as st

hw1 = st.Page("pages/HW1.py", title="HW1")
hw2 = st.Page("pages/HW2.py", title="HW2",  default=True)

# Create navigation
pg = st.navigation([hw1, hw2])


pg.run()