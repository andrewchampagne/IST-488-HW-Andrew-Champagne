import streamlit as st

lab1 = st.Page("pages/HW1.py", title="HW1")
lab2 = st.Page("pages/HW2.py", title="HW2",  default=True)

# Create navigation
pg = st.navigation([lab1, lab2])


pg.run()