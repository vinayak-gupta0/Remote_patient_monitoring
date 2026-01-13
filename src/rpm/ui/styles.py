import streamlit as st

GLOBAL_CSS = """
<style>
.block-container {
    padding-top: 2rem;
}

.rm-card {
  border-radius:18px; 
  padding:0; 
  overflow:hidden; 
  background:#ffffff;
  border:1px solid rgba(0,0,0,0.08);
  transition: box-shadow .15s ease, transform .05s ease;
}
.rm-card:hover { 
    box-shadow:0 8px 20px rgba(2,132,199,.12), 0 2px 8px rgba(2,132,199,.08); 
}

.rm-id-header {
  display:flex; 
  align-items:center; 
  justify-content:space-between;
  padding:10px 12px; 
  color:white;
}

.rm-head-normal { background:linear-gradient(90deg,#0ea5e9 0%, #38bdf8 100%); }
.rm-head-danger { background:linear-gradient(90deg,#ff0000 0%, #f87171 100%); }

.rm-id-left {
    display:flex; 
    align-items:center; 
    gap:10px;
}
.rm-avatar {
  width:36px; height:36px; border-radius:50%; background:rgba(255,255,255,.25);
  display:grid; place-items:center; font-weight:800; letter-spacing:.5px;
}
.rm-name {font-weight:700; line-height:1.05;}
.rm-sub {font-size:11px; opacity:.9}

.rm-id-body {padding:10px 12px;}
.rm-v {border:1px solid rgba(0,0,0,.06); border-radius:10px; padding:8px; background:#f8fafc; margin-bottom:8px;}
.rm-v .lab {font-size:11px; opacity:.65;}
.rm-v .val {font-size:18px; font-weight:800;}

.val-ok  { color:#065f46; }
.val-mod { color:#ffb400; }
.val-sev { color:#ff0000; }

</style>
"""


def inject() -> None:
    """Inject global CSS into the Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
