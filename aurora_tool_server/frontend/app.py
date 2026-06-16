"""AURORA Streamlit entrypoint — routes to declared pages via st.navigation."""

from __future__ import annotations

import streamlit as st

from settings_state import SHOW_AGENT_PAGE_KEY, init_navigation_state


init_navigation_state()

pages = [
    st.Page("pages/1_Pipeline_Inspector.py", title="Pipeline Inspector", default=True),
    st.Page("pages/2_Normal_Mode.py", title="Normal Mode"),
]

if st.session_state[SHOW_AGENT_PAGE_KEY]:
    pages.append(st.Page("chat.py", title="AI Agent"))

pages.append(st.Page("pages/4_Profile.py", title="Profile"))
pages.append(st.Page("pages/3_Settings.py", title="Settings"))

st.navigation(pages).run()
