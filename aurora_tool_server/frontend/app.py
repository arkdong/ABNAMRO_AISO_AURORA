"""AURORA Streamlit entrypoint — routes to declared pages via st.navigation."""

from __future__ import annotations

import streamlit as st


pages = [
    st.Page("pages/1_Pipeline_Inspector.py", title="Pipeline Inspector", default=True),
    st.Page("pages/2_Normal_Mode.py", title="Normal Mode"),
    st.Page("pages/3_Settings.py", title="Settings"),
]

st.navigation(pages).run()
