"""Shared helpers for the Streamlit pages (owner: Member 3; used by every page).

    from frontend.ui import api, no_fog
    no_fog()                                   # right after st.set_page_config
    api().get(f"{config.API_URL}/health")      # instead of httpx.get(...)
"""
from __future__ import annotations

import httpx
import streamlit as st


@st.cache_resource
def api() -> httpx.Client:
    """One HTTP client for the whole app. httpx.get() builds a new client - and its SSL context - on every call
    (~350 ms each on Windows); a shared client answers local API calls in a few milliseconds."""
    return httpx.Client(timeout=60)


def no_fog() -> None:
    """While a page reruns, Streamlit dims ("fogs") everything already on screen. Keep it fully visible: actions
    that take time show their own spinner instead."""
    st.markdown("<style>[data-stale='true'], .stale-element {opacity: 1 !important; transition: none !important;}"
                "</style>", unsafe_allow_html=True)
