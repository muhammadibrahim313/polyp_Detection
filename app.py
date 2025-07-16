"""
AI Polyp Detection System – Streamlit App
Revamped UI with modern dark/glassmorphism styling, animated header, and cleaner code
----------------------------------------------------------------------
Author: ChatGPT (revamp for user)
Date: 17‑07‑2025
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
import io
import requests
from io import BytesIO

# ────────────────────────────────────────────────────────────────────────────────
# ➤ PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏥 AI Polyp Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────────
# ➤ CUSTOM CSS (Dark + Glassmorphism)
# ────────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
:root {
  --accent1: #0ea5e9;
  --accent2: #667eea;
  --accent3: #764ba2;
  --bg1: #0d0d0d;
  --bg2: #1a1a1a;
  --glass: rgba(255,255,255,0.03);
  --text-light: #d1d5db;
}

/* --------- General --------- */
html, body, [class*="css"], .main {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: var(--text-light);
  background: var(--bg1);
}

/* Scrollbar */
::-webkit-scrollbar {width: 8px;}
::-webkit-scrollbar-track {background: var(--bg1);} 
::-webkit-scrollbar-thumb {background: var(--accent1); border-radius: 8px;}

/* -------- Hero / Header -------- */
.hero {
  background: linear-gradient(135deg, var(--accent2) 0%, var(--accent3) 100%);
  border-radius: 18px;
  padding: 3rem 2rem 4rem 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 15px 35px rgba(0,0,0,0.45);
  animation: fadeSlide 0.8s cubic-bezier(.4,.0,.2,1);
}
.hero h1 {
  color: #fff;
  font-size: 3.25rem;
  font-weight: 800;
  text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
}
.hero p {
  color: rgba(255,255,255,0.9);
  margin-top: .5rem;
  font-size: 1.1rem;
}
@keyframes fadeSlide {
  0% {opacity:0; transform:translateY(-25px);} 100% {opacity:1; transform:translateY(0);} }

/* -------- Cards & Sections -------- */
.card {
  background: var(--glass);
  backdrop-filter: blur(30px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 1.5rem;
  box-shadow: 0 10px 25px rgba(0,0,0,.35);
}
.metric {
  display:flex;flex-direction:column;align-items:center;gap:4px;
}
.metric-value {font-size:2.1rem;font-weight:700;color:var(--accent1);}
.metric-label {font-size:0.9rem;opacity:0.7;}

/* Buttons */
.stButton button {
  border:none;
  font-weight:600;
  padding: .75rem 1.75rem;
  border-radius: 10px;
  background:linear-gradient(135deg,var(--accent2),var(--accent3));
  color:#fff; transition:all .25s ease;
  box-shadow:0 4px 18px rgba(0,0,0,.45);
}
.stButton button:hover {transform:translateY(-2px);box-shadow:0 8px 25px rgba(0,0,0,.6);} 

/* Alerts */
.alert {
  padding:1.2rem 1.5rem;
  border-left:5px solid;
  border-radius:12px;
  margin-top:1rem;
}
.alert.success {background:rgba(16,185,129,.15);border-color:#10b981;color:#a7f3d0;}
.alert.danger  {background:rgba(239,68,68,.15);border-color:#ef4444;color:#fecaca;}

/* Hide Streamlit branding */
#MainMenu, footer {visibility:hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# ➤ MODEL DEFINITION (U‑Net)
# ────────────────────────────────────────────────────────────────────────────────
class UNET(nn.Module):
    def __init__(self, ch: int = 32, dropout: float = 0.1):
        super().__init__()
        pool = nn.MaxPool2d(2, 2)

        def conv_block(inp, out):
            return nn.Sequential(
                nn.Conv2d(inp, out, 3, padding=1),
                nn.BatchNorm2d(out), nn.ReLU(), nn.Dropout2d(dropout),
                nn.Conv2d(out, out, 3, padding=1),
                nn.BatchNorm2d(out), nn.ReLU(), nn.Dropout2d(dropout),
            )

        self.enc1, self.enc2, self.enc3, self.enc4 = (
            conv_block(3, ch), conv_block(ch, ch*2), conv_block(ch*2, ch*4), conv_block(ch*4, ch*8))
        self.bottle = conv_block(ch*8, ch*16)
        self.up1, self.dec1 = nn.ConvTranspose2d(ch*16, ch*8, 2, 2), conv_block(ch*16, ch*8)
        self.up2, self.dec2 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2), conv_block(ch*8, ch*4)
        self.up3, self.dec3 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2), conv_block(ch*4, ch*2)
        self.up4, self.dec4 = nn.ConvTranspose2d(ch*2, ch, 2, 2), conv_block(ch*2, ch)
        self.final = nn.Conv2d(ch, 1, 1)
        self.pool = pool

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        c5 = self.bottle(self.pool(c4))
        u6 = self.dec1(torch.cat([c4, self.up1(c5)], 1))
        u7 = self.dec2(torch.cat([c3, self.up2(u6)], 1))
        u8 = self.dec3(torch.cat([c2, self.up3(u7)], 1))
        u9 = self.dec4(torch.cat([c1, self.up4(u8)], 1))
        return self.final(u9)

# ────────────────────────────────────────────────────────────────────────────────
# ➤ SESSION STATE
# ────────────────────────────────────────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None

# ────────────────────────────────────────────────────────────────────────────────
# ➤ UTILS: Model + Prediction
# ────────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cpu')
TRANSFORM = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
    ToTensorV2()
])

@st.cache_resource(show_spinner=False)
def load_model():
    path = hf_hub_download("ibrahim313/unet-adam-diceloss", filename="pytorch_model.bin")
    model = UNET()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def predict(image: Image.Image, thresh: float=0.5):
    model: UNET = st.session_state.model
    if model is None:
        return None, 0, 0, 0
    arr = np.array(image.convert('RGB'))
    t = TRANSFORM(image=arr)['image'].unsqueeze(0).float()
    with torch.no_grad():
        out = torch.sigmoid(model(t))
        mask = (out > thresh).float().squeeze().cpu().numpy()
    polyp_px = mask.sum()
    total = mask.size
    perc = polyp_px/total*100
    # Build figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1,3, figsize=(14,4))
    titles = ['Original','Mask','Overlay']
    ax[0].imshow(arr); ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray'); ax[1].axis('off')
    ax[2].imshow(arr); ax[2].imshow(mask, cmap='Reds', alpha=0.6); ax[2].axis('off')
    for i,t in enumerate(titles): ax[i].set_title(t, color='w')
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format='png', dpi=170, bbox_inches='tight', facecolor='#0d0d0d'); buf.seek(0)
    plt.close(fig)
    return Image.open(buf), perc, int(polyp_px), total

# ────────────────────────────────────────────────────────────────────────────────
# ➤ SIDEBAR (Controls & Info)
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Controls')
    st.markdown('**Model:** U‑Net (ch=32)')
    threshold = st.slider('Detection Threshold', 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown('**Example Images**')
    ex1, ex2 = st.columns(2)
    def fetch_example(url):
        r = requests.get(url)
        return Image.open(BytesIO(r.content)) if r.ok else None
    if ex1.button('Example 1'):
        st.session_state.example = fetch_example('https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0qoxqj9q6s0835b43399p4.jpg')
    if ex2.button('Example 2'):
        st.session_state.example = fetch_example('https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0roawvklrq0799vmjorwfv.jpg')

# ────────────────────────────────────────────────────────────────────────────────
# ➤ HEADER / HERO SECTION
# ────────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="hero"><h1>AI Polyp Detection System</h1><p>Deep‑learning assisted colonoscopy analysis.</p><p style="opacity:.85">Upload an image to begin.</p></div>', unsafe_allow_html=True)

# Ensure model loaded lazily
if st.session_state.model is None:
    with st.spinner('Loading model – this happens only once...'):
        st.session_state.model = load_model()
    st.success('Model ready!')

# ────────────────────────────────────────────────────────────────────────────────
# ➤ MAIN APP LAYOUT
# ────────────────────────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1,2])

with col_upload:
    st.subheader('📤 Upload Image')
    up = st.file_uploader('Choose a colonoscopy image', type=['jpg','jpeg','png'])
    img_display = None
    if up is not None:
        img_display = Image.open(up)
    elif 'example' in st.session_state:
        img_display = st.session_state.example
    if img_display is not None:
        st.image(img_display, caption='Input Image', use_column_width=True)
    run = st.button('🔍 Analyze', use_container_width=True)

with col_results:
    st.subheader('📊 Results')
    if run and img_display is not None:
        with st.spinner('Analyzing...'):
            res_img, perc, px, total = predict(img_display, threshold)
            if res_img is None:
                st.error('Prediction failed.')
            else:
                st.session_state.results = dict(img=res_img, perc=perc, px=px, total=total, thr=threshold)

    if st.session_state.get('results'):
        r = st.session_state.results
        st.image(r['img'])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric"><span class="metric-value">{r["perc"]:.2f}%</span><span class="metric-label">Coverage</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric"><span class="metric-value">{r["px"]:,}</span><span class="metric-label">Detected Px</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric"><span class="metric-value">{r["thr"]}</span><span class="metric-label">Threshold</span></div>', unsafe_allow_html=True)

        # Alert
        if r['px'] > 100:
            st.markdown('<div class="alert danger"><h4>🚨 Polyp Detected</h4><p>Please seek medical advice.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert success"><h4>✅ No Polyp Detected</h4><p>No significant features.</p></div>', unsafe_allow_html=True)
    else:
        st.info('Upload an image and press Analyze to see results.')

# ────────────────────────────────────────────────────────────────────────────────
# ➤ FOOTER
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p style="text-align:center;font-size:.9rem;opacity:.6;">⚠️ This tool is for research/educational use only – not a diagnostic device.</p>', unsafe_allow_html=True)
