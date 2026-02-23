import streamlit as st

hw1 = st.Page("pages/HW1.py", title="HW1")
hw2 = st.Page("pages/HW2.py", title="HW2")
hw3 = st.Page("pages/HW3.py", title="HW3",  default=True)
hw4 = st.Page("pages/HW4.py", title="HW4")
hw5 = st.Page("pages/HW5.py", title="HW5")

# Create navigation
pg = st.navigation([hw1, hw2, hw3, hw4, hw5])

pg.run()