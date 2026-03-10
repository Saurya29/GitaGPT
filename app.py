import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from fpdf import FPDF, XPos, YPos

# --------------------------------------------------
# LOAD API KEY
# --------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# PAGE CONFIG  —  no sidebar
# --------------------------------------------------
st.set_page_config(
    page_title="GitaGPT — Ask Lord Krishna Anything",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# LOAD KRISHNA IMAGE
# --------------------------------------------------
krishna_b64 = None
try:
    with open("krishna_ji.jpeg", "rb") as f:
        krishna_b64 = base64.b64encode(f.read()).decode()
except:
    pass

# --------------------------------------------------
# CSS
# --------------------------------------------------
def set_css(krishna_b64=None):

    # Build the hero visual block — uses the image if available,
    # falls back to a pure CSS gradient glow
    if krishna_b64:
        hero_visual = f"""
        .krishna-visual {{
            position: relative;
            width: 100%;
            max-width: 820px;
            margin: 0 auto;
            height: 520px;
            overflow: hidden;
        }}
        /* The actual photo */
        .krishna-visual::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: url("data:image/jpeg;base64,{krishna_b64}") center top / cover no-repeat;
            mask-image: linear-gradient(
                to bottom,
                transparent 0%,
                rgba(0,0,0,0.25) 15%,
                rgba(0,0,0,0.92) 52%,
                black 70%
            );
            -webkit-mask-image: linear-gradient(
                to bottom,
                transparent 0%,
                rgba(0,0,0,0.25) 15%,
                rgba(0,0,0,0.92) 52%,
                black 70%
            );
        }}
        /* Saffron aura glow behind Krishna */
        .krishna-visual::after {{
            content: "";
            position: absolute;
            top: 5%;
            left: 50%;
            transform: translateX(-50%);
            width: 320px;
            height: 320px;
            border-radius: 50%;
            background: radial-gradient(
                circle,
                rgba(255,140,50,0.22) 0%,
                rgba(212,175,55,0.12) 40%,
                transparent 70%
            );
            mix-blend-mode: screen;
            animation: auraPulse 4s ease-in-out infinite;
        }}
        /* Floating particle dots */
        .krishna-visual .particle {{
            position: absolute;
            border-radius: 50%;
            background: rgba(212,175,55,0.6);
            animation: floatUp linear infinite;
            pointer-events: none;
        }}
        """
    else:
        hero_visual = """
        .krishna-visual {
            position: relative;
            width: 100%;
            max-width: 820px;
            margin: 0 auto;
            height: 340px;
            background: radial-gradient(ellipse at 50% 30%,
                rgba(255,140,50,0.18) 0%,
                rgba(212,175,55,0.08) 40%,
                transparent 70%);
        }
        .krishna-visual::after { display: none; }
        """

    st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

/* ── VARIABLES ── */
:root {{
    --bg:           #070711;
    --bg2:          #0d0d1a;
    --surface:      #0f0f1c;
    --surface2:     #161625;
    --surface3:     #1c1c2e;
    --border:       rgba(255,255,255,0.06);
    --border-soft:  rgba(255,255,255,0.04);
    --border-gold:  rgba(212,175,55,0.22);
    --saffron:      #E8622A;
    --saffron-l:    #FF7A45;
    --saffron-dim:  rgba(232,98,42,0.10);
    --saffron-glow: rgba(232,98,42,0.22);
    --gold:         #C9A84C;
    --gold-dim:     rgba(201,168,76,0.13);
    --cream:        #EDE8DE;
    --cream2:       #F5F0E8;
    --muted:        rgba(237,232,222,0.38);
    --muted2:       rgba(237,232,222,0.55);
    --online:       #34D399;
    --r:            14px;
    --r-sm:         10px;
}}

/* ── RESET ── */
html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    background: linear-gradient(160deg, var(--bg) 0%, var(--bg2) 100%) !important;
    color: var(--cream) !important;
    -webkit-font-smoothing: antialiased;
}}

/* hide sidebar toggle & default chrome */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu, footer, header,
.stDeployButton {{ display: none !important; }}

.block-container {{
    padding: 0 !important;
    max-width: 820px !important;
}}

/* ══════════════════════════════════════
   NAV BAR
══════════════════════════════════════ */
.topnav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.95rem 2.2rem;
    border-bottom: 1px solid var(--border);
    background: rgba(7,7,17,0.97);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 200;
}}
.nav-logo {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--cream2);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.nav-logo em {{ color: var(--saffron-l); font-style: normal; }}
.nav-links {{
    display: flex;
    gap: 2rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.02em;
}}
.nav-links span {{
    cursor: pointer;
    transition: color 0.2s ease;
    padding-bottom: 2px;
    border-bottom: 1px solid transparent;
}}
.nav-links span:hover {{
    color: var(--cream);
    border-bottom-color: var(--saffron-l);
}}

/* ══════════════════════════════════════
   KRISHNA VISUAL HERO
══════════════════════════════════════ */
{hero_visual}

/* Text layer sits on top of the image */
.hero-text {{
    position: absolute;
    bottom: 0;
    left: 0; right: 0;
    padding: 2.5rem 2.5rem 3rem;
    text-align: center;
    z-index: 10;
    animation: heroFadeIn 1s ease both;
}}
.hero-eyebrow {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--saffron-l);
    margin-bottom: 0.9rem;
    display: block;
    opacity: 0.9;
}}
.hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -0.01em;
    color: var(--cream2);
    margin-bottom: 0.75rem;
    text-shadow: 0 2px 32px rgba(0,0,0,0.7), 0 0 60px rgba(0,0,0,0.5);
}}
.hero-title .g {{
    background: linear-gradient(110deg, var(--saffron-l) 0%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-sub {{
    font-family: 'Crimson Pro', serif;
    font-size: 1.12rem;
    color: rgba(237,232,222,0.72);
    max-width: 420px;
    margin: 0 auto 1.6rem;
    line-height: 1.7;
    font-weight: 400;
}}

/* ══════════════════════════════════════
   AVATAR STATUS BADGE
══════════════════════════════════════ */
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(7,7,17,0.82);
    border: 1px solid rgba(201,168,76,0.28);
    border-radius: 40px;
    padding: 0.5rem 1.3rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--cream);
    backdrop-filter: blur(12px);
    letter-spacing: 0.02em;
    box-shadow:
        0 4px 24px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.04);
}}
.dot-live {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--online);
    box-shadow: 0 0 10px var(--online), 0 0 4px var(--online);
    animation: blink 2.4s ease infinite;
    flex-shrink: 0;
}}

/* ══════════════════════════════════════
   STATS ROW
══════════════════════════════════════ */
.stats-row {{
    display: flex;
    justify-content: center;
    gap: 1px;
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
    background: var(--border);
    margin: 0 1.8rem;
}}
.stat {{
    flex: 1;
    padding: 1.1rem 0.5rem;
    text-align: center;
    background: var(--surface);
    transition: background 0.2s ease;
}}
.stat:hover {{
    background: var(--surface2);
}}
.stat-v {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--cream2);
    display: block;
    letter-spacing: -0.01em;
}}
.stat-l {{
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
    display: block;
    font-weight: 500;
}}

/* ══════════════════════════════════════
   INTRO FORM
══════════════════════════════════════ */
.form-shell {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(255,255,255,0.05);
    border-radius: var(--r);
    padding: 2.2rem 2.4rem 2rem;
    margin: 2rem 1.8rem 0;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.03) inset,
        0 20px 60px rgba(0,0,0,0.3);
}}
.form-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--cream2);
    margin-bottom: 0.3rem;
    letter-spacing: 0.01em;
}}
.form-sub {{
    font-family: 'Crimson Pro', serif;
    font-size: 0.94rem;
    color: var(--muted);
    font-style: italic;
    margin-bottom: 1.6rem;
    line-height: 1.5;
}}
[data-testid="stForm"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}}

/* ══════════════════════════════════════
   INPUTS
══════════════════════════════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background: var(--surface2) !important;
    color: var(--cream2) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px var(--saffron-dim) !important;
}}
.stTextInput label, .stNumberInput label {{
    color: var(--muted2) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}}

/* ══════════════════════════════════════
   BUTTONS
══════════════════════════════════════ */
.stButton > button {{
    background: var(--saffron) !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.02em !important;
    transition: all .22s ease !important;
    box-shadow: 0 2px 20px var(--saffron-glow) !important;
}}
.stButton > button:hover {{
    background: var(--saffron-l) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 28px var(--saffron-glow) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ══════════════════════════════════════
   CHAT AREA WRAPPER
══════════════════════════════════════ */
.chat-wrap {{
    padding: 0 1.4rem;
}}

/* Krishna floating avatar chip above chat */
.krishna-chip {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1.2rem;
    background: var(--surface);
    border: 1px solid var(--border-gold);
    border-radius: var(--r);
    margin: 1.4rem 1.4rem 0.8rem;
    box-shadow: 0 0 30px rgba(212,175,55,0.06);
}}
.chip-avatar {{
    width: 44px; height: 44px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid var(--saffron);
    box-shadow: 0 0 0 4px var(--saffron-dim);
    flex-shrink: 0;
    background: linear-gradient(135deg, #FF6B35, #D4AF37);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
}}
.chip-info {{ flex: 1; }}
.chip-name {{
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--cream);
}}
.chip-status {{
    font-size: 0.73rem;
    color: var(--online);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.15rem;
}}
.chip-shloka {{
    font-family: 'Crimson Pro', serif;
    font-size: 0.8rem;
    color: rgba(212,175,55,0.7);
    font-style: italic;
    text-align: right;
    line-height: 1.5;
    max-width: 200px;
}}

/* ══════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════ */
.stChatInput > div {{
    background: var(--surface) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin: 0 1.8rem !important;
    transition: border-color .2s, box-shadow .2s !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
}}
.stChatInput > div:focus-within {{
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px var(--saffron-dim), 0 4px 24px rgba(0,0,0,0.2) !important;
}}
.stChatInput textarea {{
    background: transparent !important;
    color: var(--cream2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}}
.stChatInput textarea::placeholder {{
    color: var(--muted) !important;
    font-style: italic;
}}

/* ══════════════════════════════════════
   CHAT BUBBLES
══════════════════════════════════════ */
[data-testid="stChatMessage"] {{
    border-radius: var(--r) !important;
    margin: 0.5rem 1.8rem !important;
    border: 1px solid var(--border) !important;
    animation: fadeUp .3s ease both;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: linear-gradient(135deg, #111124 0%, #0e0e1e 100%) !important;
    border-color: rgba(255,255,255,0.05) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    background: linear-gradient(135deg, #0c0c18 0%, #0a0a14 100%) !important;
    border-color: rgba(201,168,76,0.16) !important;
    box-shadow:
        0 0 0 1px rgba(201,168,76,0.06),
        0 8px 32px rgba(0,0,0,0.25) !important;
}}
[data-testid="stChatMessage"] p {{
    font-family: 'Crimson Pro', serif !important;
    font-size: 1.08rem !important;
    line-height: 1.82 !important;
    color: var(--cream) !important;
}}

/* ══════════════════════════════════════
   FLOATING ACTION BUTTONS
══════════════════════════════════════ */
.fab-row {{
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem 0;
}}
.fab {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.75rem;
    color: var(--muted);
    cursor: pointer;
    transition: all .2s ease;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}}
.fab:hover {{
    border-color: var(--saffron);
    color: var(--saffron);
}}

/* Download button */
.stDownloadButton > button {{
    background: var(--surface2) !important;
    color: var(--muted2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.9rem !important;
    box-shadow: none !important;
    transition: all .2s ease !important;
}}
.stDownloadButton > button:hover {{
    border-color: var(--saffron) !important;
    color: var(--saffron-l) !important;
    transform: none !important;
}}

/* ══════════════════════════════════════
   FOOTER
══════════════════════════════════════ */
.gita-footer {{
    text-align: center;
    padding: 2.5rem 1.8rem 2rem;
    font-size: 0.76rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    line-height: 2.1;
    letter-spacing: 0.02em;
}}

/* ══════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════ */
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes heroFadeIn {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes blink {{
    0%,100% {{ opacity:1; }}
    50%      {{ opacity:0.3; }}
}}
@keyframes auraPulse {{
    0%,100% {{ opacity:.65; transform:translateX(-50%) scale(1); }}
    50%      {{ opacity:1;   transform:translateX(-50%) scale(1.15); }}
}}
@keyframes floatUp {{
    0%   {{ transform:translateY(0) scale(1);   opacity:.8; }}
    100% {{ transform:translateY(-120px) scale(0); opacity:0; }}
}}

</style>
""", unsafe_allow_html=True)


set_css(krishna_b64)

# --------------------------------------------------
# NAV BAR
# --------------------------------------------------
st.markdown("""
<div class="topnav">
    <div class="nav-logo">🕉️ &nbsp;<em>GitaGPT</em></div>
    <div class="nav-links">
        <span>Chat</span>
        <span>Library</span>
        <span>About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# API KEY CHECK
# --------------------------------------------------
if not GROQ_API_KEY:
    st.markdown("""
    <div style="padding:4rem 2rem;text-align:center;">
        <div style="font-size:2rem;margin-bottom:1rem;">🔑</div>
        <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">
            API Key Missing
        </div>
        <div style="font-size:0.88rem;color:rgba(245,240,232,0.45);">
            Add <code>GROQ_API_KEY</code> to your <code>.env</code> file
            or Streamlit Cloud → Secrets.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# --------------------------------------------------
# VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def create_vectorstore():
    loader   = PyPDFLoader("gita_book.pdf")
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs     = splitter.split_documents(pages)
    emb      = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embedding=emb, persist_directory="./gita_chroma")

vectorstore = create_vectorstore()
qa_chain    = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages"     not in st.session_state: st.session_state.messages     = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --------------------------------------------------
# INTRO / NAME GATE
# --------------------------------------------------
if "name" not in st.session_state:

    # ── HERO: Krishna image fades into the headline ──
    st.markdown("""
    <div class="krishna-visual">
        <div class="hero-text">
            <span class="hero-eyebrow">Ancient Wisdom &nbsp;•&nbsp; Modern Guidance</span>
            <div class="hero-title">
                Ask <span class="g">Lord Krishna</span><br>Anything
            </div>
            <div class="hero-sub">
                Personalized guidance inspired by the Bhagavad Gita<br>for life's deepest questions.
            </div>
            <div class="status-badge">
                <div class="dot-live"></div>
                Lord Krishna AI &nbsp;·&nbsp; Always Online
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stats-row">
        <div class="stat"><span class="stat-v">700+</span><span class="stat-l">Shlokas Indexed</span></div>
        <div class="stat"><span class="stat-v">100%</span><span class="stat-l">Private & Secure</span></div>
        <div class="stat"><span class="stat-v">RAG</span><span class="stat-l">AI Powered</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Form
    st.markdown("""
    <div class="form-shell">
        <div class="form-title">Begin your session</div>
        <div class="form-sub">No account required. Your conversations are private and never stored.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("intro"):
        name   = st.text_input("Your Name", placeholder="e.g. Arjuna")
        age    = st.number_input("Your Age", min_value=1, max_value=120, value=25)
        submit = st.form_submit_button("🕉️  Begin Conversation")
        if submit:
            if name.strip():
                st.session_state.name = name.strip()
                st.session_state.age  = age
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")

# --------------------------------------------------
# MAIN CHAT
# --------------------------------------------------
else:
    name = st.session_state.name
    age  = st.session_state.age

    # ── Compact Krishna visual strip (chat page) ──
    # Shows the image cropped to face region with glow overlay
    if krishna_b64:
        st.markdown(f"""
        <div style="
            width:100%;
            height:260px;
            position:relative;
            overflow:hidden;
        ">
            <!-- photo -->
            <div style="
                position:absolute; inset:0;
                background: url('data:image/jpeg;base64,{krishna_b64}') center 15% / cover no-repeat;
                mask-image: linear-gradient(to bottom,
                    transparent 0%, rgba(0,0,0,0.08) 8%,
                    rgba(0,0,0,0.85) 58%, black 78%);
                -webkit-mask-image: linear-gradient(to bottom,
                    transparent 0%, rgba(0,0,0,0.08) 8%,
                    rgba(0,0,0,0.85) 58%, black 78%);
            "></div>
            <!-- saffron aura glow -->
            <div style="
                position:absolute; top:0; left:50%; transform:translateX(-50%);
                width:280px; height:220px; border-radius:50%;
                background: radial-gradient(circle,
                    rgba(255,140,50,0.2) 0%,
                    rgba(212,175,55,0.1) 45%,
                    transparent 70%);
                mix-blend-mode:screen;
                animation: auraPulse 4s ease-in-out infinite;
                pointer-events:none;
            "></div>
            <!-- text overlay -->
            <div style="
                position:absolute; bottom:0; left:0; right:0;
                padding:1rem 1.6rem 1.2rem;
                display:flex; align-items:flex-end;
                justify-content:space-between;
            ">
                <div>
                    <div style="font-size:0.65rem;font-weight:600;
                                letter-spacing:0.2em;text-transform:uppercase;
                                color:#FF6B35;margin-bottom:0.3rem;">
                        श्रीमद्भगवद्गीता
                    </div>
                    <div style="font-size:1.4rem;font-weight:700;
                                letter-spacing:-0.02em;color:#F5F0E8;">
                        Lord Krishna AI
                    </div>
                </div>
                <div style="
                    display:flex; align-items:center; gap:0.45rem;
                    background:rgba(10,10,18,0.7);
                    border:1px solid rgba(212,175,55,0.25);
                    border-radius:20px; padding:0.35rem 0.9rem;
                    font-size:0.72rem; color:#22C55E; font-weight:500;
                    backdrop-filter:blur(8px);
                ">
                    <div style="width:6px;height:6px;border-radius:50%;
                                background:#22C55E;box-shadow:0 0 6px #22C55E;
                                animation:blink 2s ease infinite;"></div>
                    Always online
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Action row (clear + download) ──
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown(f"""
        <div style="padding:0.6rem 1.4rem 0;font-size:0.85rem;color:rgba(245,240,232,0.5);">
            🙏 &nbsp;Namaste, <strong style="color:#F5F0E8;">{name}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Clear"):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()
    with col3:
        hist = st.session_state.get("chat_history", [])
        if hist:
            def make_pdf(history, seeker):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", "B", 18)
                pdf.set_text_color(180, 100, 0)
                pdf.cell(0, 12, "GitaGPT — Dialogue Transcript",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
                pdf.set_font("helvetica", "", 10)
                pdf.set_text_color(80, 40, 0)
                pdf.cell(0, 8, f"Seeker: {seeker}",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
                pdf.ln(4)
                for i, (q, a) in enumerate(history, 1):
                    pdf.set_font("helvetica", "B", 10)
                    pdf.set_text_color(0, 77, 97)
                    pdf.multi_cell(0, 7,
                        f"[{i}] You: {q.encode('latin-1','replace').decode('latin-1')}",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font("helvetica", "", 10)
                    pdf.set_text_color(59, 0, 0)
                    clean = a.replace("🪷 **Lord Krishna speaks —**","").strip()
                    pdf.multi_cell(0, 7,
                        f"Krishna: {clean.encode('latin-1','replace').decode('latin-1')}",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(3)
                return bytes(pdf.output())

            st.download_button(
                "📄 Save",
                data=make_pdf(hist, name),
                file_name=f"GitaGPT_{name}.pdf",
                mime="application/pdf"
            )

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ──
    user_input = st.chat_input("Type a message…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Krishna is reflecting…"):
                try:
                    prompt = f"""You are Lord Krishna speaking to {name} (age {age}).
Use wisdom from the Bhagavad Gita. Structure your response:
1. Relevant Sanskrit shloka (chapter/verse)
2. English translation
3. Practical spiritual guidance

Keep the tone calm, wise, and compassionate. Never say you are an AI.
Question: {user_input}"""
                    response = qa_chain.run(prompt)
                    final    = f"🪷 **Lord Krishna speaks —**\n\n{response}"
                except Exception as e:
                    final = f"*The divine connection wavered. Please try again.*\n\n`{e}`"

                st.markdown(final)

        st.session_state.messages.append({"role": "assistant", "content": final})
        st.session_state.chat_history.append((user_input, final))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="gita-footer">
    🕉️ &nbsp; हरे कृष्ण हरे कृष्ण &nbsp; 🕉️<br>
    <strong>GitaGPT</strong> &nbsp;·&nbsp; Created by Saurya Raj
    &nbsp;·&nbsp; Powered by Groq + LangChain
</div>
""", unsafe_allow_html=True)