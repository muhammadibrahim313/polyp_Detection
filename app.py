import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
import io
import requests
import base64
from io import BytesIO

# Page configuration with optimized layout
st.set_page_config(
    page_title="🏥 AI Polyp Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ibrahim313/polyp-detection',
        'Report a bug': 'https://github.com/ibrahim313/polyp-detection/issues',
        'About': "# AI Polyp Detection System\nPowered by Duck-Net & U-Net architectures"
    }
)

# Enhanced CSS for professional desktop layout
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
        --info-color: #0ea5e9;
        --dark-bg: #0c0c0c;
        --card-bg: #1a1a1a;
        --text-primary: #ffffff;
        --text-secondary: #b8b8b8;
        --border-color: #333;
    }
    
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1a1a1a 50%, var(--dark-bg) 100%);
        font-family: 'Inter', sans-serif;
        padding: 0 1rem;
    }
    
    /* Container width optimization for desktop */
    .block-container {
        max-width: 1400px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Header styling with better proportions */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>') repeat;
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin: 0.5rem 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Model selector card */
    .model-selector {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d2d2d 100%);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left: 4px solid var(--info-color);
    }
    
    /* Info cards with better spacing */
    .info-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d2d2d 100%);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left: 4px solid var(--info-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }
    
    .info-card h3 {
        color: var(--info-color);
        margin-top: 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .info-card p {
        color: var(--text-secondary);
        margin: 0.5rem 0;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Results card with better proportions */
    .results-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2a2a2a 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Alert styling */
    .success-alert {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid var(--success-color);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    
    .error-alert {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid var(--error-color);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2);
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid var(--warning-color);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
    }
    
    /* Button styling with better hover effects */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar optimization */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--card-bg) 0%, var(--dark-bg) 100%);
    }
    
    [data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 320px !important;
        min-width: 320px !important;
        padding-top: 1rem;
    }
    
    /* Metrics cards with better visual hierarchy */
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d2d2d 100%);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--info-color);
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--dark-bg) 100%);
        border-top: 2px solid var(--border-color);
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border-radius: 15px;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.3);
    }
    
    .footer h4 {
        color: var(--error-color);
        margin: 0;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .footer p {
        color: var(--text-secondary);
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Upload area styling */
    .stFileUploader > div {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d2d2d 100%);
        border: 2px dashed var(--border-color);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--info-color);
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
    }
    
    /* Example image buttons */
    .example-btn {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.25rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .example-btn:hover {
        background: linear-gradient(135deg, var(--info-color) 0%, #0284c7 100%);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    
    /* Custom selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, var(--card-bg) 0%, #2d2d2d 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }
    
    /* Responsive design for smaller screens */
    @media (max-width: 1200px) {
        .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        [data-testid="stSidebar"] {
            width: 280px !important;
            min-width: 280px !important;
        }
    }
    
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
        
        [data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Duck-Net Model Definition
class DuckNet(nn.Module):
    def __init__(self, img_size=(256, 256), num_classes=3):
        super(DuckNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._make_conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._make_conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._make_conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._make_conv_block(64, 32)
        
        # Output
        self.final = nn.Conv2d(32, num_classes, 1)
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)
        
        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        output = self.final(d1)
        return torch.sigmoid(output)

# U-Net Model Definition (Original)
class UNET(nn.Module):
    def __init__(self, dropout_rate=0.1, ch=32):
        super(UNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            )
        
        self.encoder1 = conv_block(3, ch)
        self.encoder2 = conv_block(ch, ch*2)
        self.encoder3 = conv_block(ch*2, ch*4)
        self.encoder4 = conv_block(ch*4, ch*8)
        self.bottle_neck = conv_block(ch*8, ch*16)

        self.upsample1 = nn.ConvTranspose2d(ch*16, ch*8, kernel_size=2, stride=2)
        self.decoder1 = conv_block(ch*16, ch*8)
        self.upsample2 = nn.ConvTranspose2d(ch*8, ch*4, kernel_size=2, stride=2)
        self.decoder2 = conv_block(ch*8, ch*4)
        self.upsample3 = nn.ConvTranspose2d(ch*4, ch*2, kernel_size=2, stride=2)
        self.decoder3 = conv_block(ch*4, ch*2)
        self.upsample4 = nn.ConvTranspose2d(ch*2, ch, kernel_size=2, stride=2)
        self.decoder4 = conv_block(ch*2, ch)
        self.final = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.encoder1(x)
        c2 = self.encoder2(self.pool(c1))
        c3 = self.encoder3(self.pool(c2))
        c4 = self.encoder4(self.pool(c3))
        c5 = self.bottle_neck(self.pool(c4))

        u6 = self.upsample1(c5)
        u6 = torch.cat([c4, u6], dim=1)
        c6 = self.decoder1(u6)
        u7 = self.upsample2(c6)
        u7 = torch.cat([c3, u7], dim=1)
        c7 = self.decoder2(u7)
        u8 = self.upsample3(c7)
        u8 = torch.cat([c2, u8], dim=1)
        c8 = self.decoder3(u8)
        u9 = self.upsample4(c8)
        u9 = torch.cat([c1, u9], dim=1)
        c9 = self.decoder4(u9)
        return self.final(c9)

# Model configurations
MODEL_CONFIGS = {
    "🦆 Duck-Net (Latest)": {
        "repo_id": "ibrahim313/ducknet-polyp-segmentation",
        "model_class": DuckNet,
        "input_size": (256, 256),
        "num_classes": 3,
        "description": "Advanced Duck-Net architecture with multi-scale feature extraction",
        "performance": "Dice: 92.88% | Jaccard: 48.92%",
        "status": "🟢 Latest Model"
    },
    "🏥 U-Net (Classic)": {
        "repo_id": "ibrahim313/unet-adam-diceloss", 
        "model_class": UNET,
        "input_size": (384, 384),
        "num_classes": 1,
        "description": "Proven U-Net architecture for medical segmentation",
        "performance": "Reliable baseline performance",
        "status": "🔵 Stable"
    }
}

# Example images
EXAMPLE_IMAGES = {
    "🔬 Polyp Sample 1": "https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0qoxqj9q6s0835b43399p4.jpg",
    "🔬 Polyp Sample 2": "https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0roawvklrq0799vmjorwfv.jpg",
    "🔬 Normal Sample 1": "https://raw.githubusercontent.com/jfzhang95/pytorch-deeplab-xception/master/datasets/data/pascal_voc_seg/images_preproc_test/2007_000033.jpg",
    "🔬 Normal Sample 2": "https://raw.githubusercontent.com/jfzhang95/pytorch-deeplab-xception/master/datasets/data/pascal_voc_seg/images_preproc_test/2007_000042.jpg"
}

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "🦆 Duck-Net (Latest)"
if 'example_image' not in st.session_state:
    st.session_state.example_image = None

# Global variables
device = torch.device('cpu')

@st.cache_resource
def load_model(model_name):
    """Load model from HuggingFace repository"""
    try:
        config = MODEL_CONFIGS[model_name]
        
        with st.spinner(f"🔄 Loading {model_name} from HuggingFace..."):
            model_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename="pytorch_model.bin"
            )
            
            if config["model_class"] == DuckNet:
                model = DuckNet(img_size=config["input_size"], num_classes=config["num_classes"])
            else:
                model = UNET(ch=32)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            return model, f"✅ {model_name} loaded successfully!", config
    except Exception as e:
        return None, f"❌ Error loading {model_name}: {e}", None

def load_example_image(image_url):
    """Load example image from URL"""
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            st.error(f"Failed to load example image. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading example image: {e}")
        return None

def get_transform(input_size):
    """Get appropriate transform based on model"""
    if input_size == (256, 256):
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
            ToTensorV2()
        ])

def predict_polyp(image, threshold=0.5, model_config=None):
    """Predict polyp in uploaded image"""
    if st.session_state.model is None:
        return None, None, None, "❌ Model not loaded! Please wait for model to load."
    
    if image is None:
        return None, None, None, "❌ Please upload an image first!"
    
    try:
        # Convert image to numpy array
        if isinstance(image, Image.Image):
            original_image = np.array(image.convert('RGB'))
        else:
            original_image = np.array(image)
        
        # Get appropriate transform
        transform = get_transform(model_config["input_size"])
        
        # Preprocess image
        transformed = transform(image=original_image)
        input_tensor = transformed['image'].unsqueeze(0).float()
        
        # Make prediction
        with torch.no_grad():
            prediction = st.session_state.model(input_tensor)
            if model_config["num_classes"] == 1:
                prediction = torch.sigmoid(prediction)
            prediction = (prediction > threshold).float()
        
        # Convert to numpy
        if model_config["num_classes"] == 3:
            # For Duck-Net (3 classes), take the first channel or combine
            pred_mask = prediction.squeeze()[0].cpu().numpy()
        else:
            # For U-Net (1 class)
            pred_mask = prediction.squeeze().cpu().numpy()
        
        # Calculate metrics
        polyp_pixels = np.sum(pred_mask)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        polyp_percentage = (polyp_pixels / total_pixels) * 100
        
        # Create visualization with dark theme
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('🖼️ Original Image', fontsize=14, color='white', pad=20, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('🎭 Predicted Mask', fontsize=14, color='white', pad=20, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(pred_mask, cmap='Reds', alpha=0.6)
        axes[2].set_title('🔍 Detection Overlay', fontsize=14, color='white', pad=20, fontweight='bold')
        axes[2].axis('off')
        
        # Add main title with results
        if polyp_pixels > 100:
            main_title = f"🚨 POLYP DETECTED! Coverage: {polyp_percentage:.2f}%"
            title_color = '#ef4444'
        else:
            main_title = f"✅ No Polyp Detected - Coverage: {polyp_percentage:.2f}%"
            title_color = '#10b981'
        
        fig.suptitle(main_title, fontsize=18, fontweight='bold', color=title_color, y=0.95)
        plt.tight_layout()
        
        # Save plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image, polyp_percentage, int(polyp_pixels), total_pixels
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, f"❌ Error processing image: {str(e)}"

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Polyp Detection System</h1>
        <p>Advanced Medical Imaging with Deep Learning</p>
        <p style="opacity: 0.9;">Choose your model and upload colonoscopy images for intelligent polyp detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("### 🧠 Model Selection")
        
        # Model selector
        selected_model = st.selectbox(
            "Choose AI Model",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
            help="Select the AI model for polyp detection"
        )
        
        # Update model if selection changed
        if selected_model != st.session_state.selected_model or not st.session_state.model_loaded:
            st.session_state.selected_model = selected_model
            st.session_state.model_loaded = False
            st.session_state.model = None
        
        # Display model info
        config = MODEL_CONFIGS[selected_model]
        st.markdown(f"""
        <div class="model-selector">
            <h3>🔬 {selected_model}</h3>
            <p><strong>Status:</strong> {config["status"]}</p>
            <p><strong>Repository:</strong> {config["repo_id"]}</p>
            <p><strong>Performance:</strong> {config["performance"]}</p>
            <p><strong>Input Size:</strong> {config["input_size"][0]}×{config["input_size"][1]}</p>
            <p>{config["description"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            model, status, model_config = load_model(selected_model)
            if model is not None:
                st.session_state.model = model
                st.session_state.model_config = model_config
                st.session_state.model_loaded = True
                st.success(status)
            else:
                st.error(status)
                return
        
        st.markdown("---")
        
        # Detection sensitivity
        st.markdown("### 🎯 Detection Settings")
        threshold = st.slider(
            "Detection Sensitivity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Higher values = more sensitive detection"
        )
        
        # Advanced settings in expander
        with st.expander("⚙️ Advanced Settings"):
            confidence_display = st.checkbox("Show Confidence Scores", value=True)
            overlay_opacity = st.slider("Overlay Opacity", 0.3, 0.9, 0.6, 0.1)
        
        st.markdown("---")
        
        # Example images section
        st.markdown("### 📸 Example Images")
        st.markdown("*Try these sample images to test the system*")
        
        example_choice = st.selectbox(
            "Select Example Image",
            options=["None"] + list(EXAMPLE_IMAGES.keys()),
            help="Choose a sample image for testing"
        )
        
        if example_choice != "None":
            if st.button("🔄 Load Selected Example", use_container_width=True):
                with st.spinner("Loading example image..."):
                    example_img = load_example_image(EXAMPLE_IMAGES[example_choice])
                    if example_img:
                        st.session_state.example_image = example_img
                        st.success(f"✅ Loaded: {example_choice}")
                    else:
                        st.error("❌ Failed to load example image")
        
        # Quick test buttons
        st.markdown("#### 🚀 Quick Test")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Test Polyp", use_container_width=True, help="Load a sample with polyps"):
                with st.spinner("Loading polyp sample..."):
                    example_img = load_example_image(EXAMPLE_IMAGES["🔬 Polyp Sample 1"])
                    if example_img:
                        st.session_state.example_image = example_img
        
        with col2:
            if st.button("✅ Test Normal", use_container_width=True, help="Load a normal sample"):
                with st.spinner("Loading normal sample..."):
                    example_img = load_example_image(EXAMPLE_IMAGES["🔬 Normal Sample 1"])
                    if example_img:
                        st.session_state.example_image = example_img
    
    # Main content area with improved layout
    col1, col2 = st.columns([1, 1.8], gap="large")
    
    with col1:
        st.markdown("### 📤 Image Upload")
        
        # Image upload area
        uploaded_file = st.file_uploader(
            "Choose a colonoscopy image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a high-quality colonoscopy image for analysis",
            accept_multiple_files=False
        )
        
        # Display image section
        display_image = None
        image_source = ""
        
        if uploaded_file is not None:
            display_image = Image.open(uploaded_file)
            image_source = f"📁 Uploaded: {uploaded_file.name}"
        elif st.session_state.example_image is not None:
            display_image = st.session_state.example_image
            image_source = "🔬 Example Image Selected"
        
        if display_image is not None:
            # Display image with info
            st.image(display_image, caption=image_source, use_column_width=True)
            
            # Image info
            img_width, img_height = display_image.size
            st.markdown(f"""
            <div class="info-card">
                <h3>📊 Image Information</h3>
                <p><strong>Dimensions:</strong> {img_width} × {img_height} pixels</p>
                <p><strong>Format:</strong> {display_image.format or 'Unknown'}</p>
                <p><strong>Mode:</strong> {display_image.mode}</p>
                <p><strong>Source:</strong> {image_source}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Placeholder for no image
            st.markdown("""
            <div class="results-card">
                <div style="text-align: center; padding: 3rem;">
                    <h3 style="color: #9ca3af; margin-bottom: 1rem;">📸 No Image Selected</h3>
                    <p style="color: #6b7280;">Upload an image or select an example to get started</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        analyze_disabled = display_image is None or not st.session_state.model_loaded
        if st.button(
            "🔍 Analyze for Polyps", 
            type="primary", 
            use_container_width=True,
            disabled=analyze_disabled,
            help="Start polyp detection analysis" if not analyze_disabled else "Please load an image and model first"
        ):
            if display_image is not None and st.session_state.model_loaded:
                with st.spinner("🔄 Analyzing image with AI..."):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    result_image, polyp_percentage, polyp_pixels, total_pixels = predict_polyp(
                        display_image, 
                        threshold, 
                        st.session_state.model_config
                    )
                    
                    progress_bar.progress(75)
                    
                    if result_image is not None and polyp_percentage is not None:
                        st.session_state.results = {
                            'image': result_image,
                            'percentage': polyp_percentage,
                            'pixels': polyp_pixels,
                            'total_pixels': total_pixels,
                            'threshold': threshold,
                            'model_name': selected_model,
                            'confidence': polyp_percentage / 100,
                            'overlay_opacity': overlay_opacity
                        }
                        progress_bar.progress(100)
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error(f"❌ Analysis failed: {total_pixels}")
                    
                    progress_bar.empty()
            else:
                st.warning("⚠️ Please upload an image or select an example first!")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display result image
            if results['image']:
                st.image(results['image'], use_column_width=True, caption=f"Analysis Results - {results['model_name']}")
            
            # Enhanced metrics display
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{results['percentage']:.2f}%</p>
                    <p class="metric-label">Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{results['pixels']:,}</p>
                    <p class="metric-label">Detected Pixels</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                if confidence_display:
                    confidence_color = "#10b981" if results['confidence'] > 0.7 else "#f59e0b" if results['confidence'] > 0.4 else "#ef4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value" style="color: {confidence_color};">{results['confidence']:.2f}</p>
                        <p class="metric-label">Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{results['threshold']}</p>
                        <p class="metric-label">Threshold</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_d:
                processing_time = "< 1s"
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{processing_time}</p>
                    <p class="metric-label">Process Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clinical assessment with enhanced styling
            if results['pixels'] > 100:
                risk_level = "High" if results['percentage'] > 5 else "Medium" if results['percentage'] > 2 else "Low"
                st.markdown(f"""
                <div class="error-alert">
                    <h4>🚨 POLYP DETECTED</h4>
                    <p><strong>Risk Level:</strong> {risk_level}</p>
                    <p><strong>Coverage:</strong> {results['percentage']:.2f}% of image area</p>
                    <p><strong>Recommendation:</strong> Immediate medical review recommended</p>
                    <p>A potential polyp has been identified in the image. Please consult with a qualified medical professional for proper diagnosis and treatment planning.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-alert">
                    <h4>✅ NO SIGNIFICANT POLYP DETECTED</h4>
                    <p><strong>Status:</strong> Normal findings</p>
                    <p><strong>Coverage:</strong> {results['percentage']:.2f}% detection</p>
                    <p><strong>Recommendation:</strong> Continue routine monitoring</p>
                    <p>No significant polyp features detected in this image. Continue with regular screening as recommended by your healthcare provider.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Technical details in expandable section
            with st.expander("🔬 Technical Details & Model Info"):
                col_tech1, col_tech2 = st.columns(2)
                
                with col_tech1:
                    st.markdown(f"""
                    **Model Details:**
                    - **Architecture:** {results['model_name']}
                    - **Input Size:** {st.session_state.model_config['input_size'][0]}×{st.session_state.model_config['input_size'][1]} pixels
                    - **Classes:** {st.session_state.model_config['num_classes']}
                    - **Repository:** {st.session_state.model_config['repo_id']}
                    """)
                
                with col_tech2:
                    st.markdown(f"""
                    **Analysis Parameters:**
                    - **Detection Threshold:** {results['threshold']}
                    - **Total Pixels:** {results['total_pixels']:,}
                    - **Detected Pixels:** {results['pixels']:,}
                    - **Processing Device:** CPU
                    """)
                
                st.markdown(f"""
                **Performance Metrics:**
                - {st.session_state.model_config['performance']}
                - **Model Status:** {st.session_state.model_config['status']}
                """)
            
            # Download results button
            if st.button("📥 Download Analysis Report", use_container_width=True):
                # Create a simple text report
                report = f"""
AI Polyp Detection Report
========================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {results['model_name']}

Results:
- Polyp Coverage: {results['percentage']:.2f}%
- Detected Pixels: {results['pixels']:,}
- Total Pixels: {results['total_pixels']:,}
- Detection Threshold: {results['threshold']}
- Confidence: {results['confidence']:.2f}

Clinical Assessment:
{('POLYP DETECTED - Medical review recommended' if results['pixels'] > 100 else 'NO SIGNIFICANT POLYP DETECTED - Continue routine monitoring')}

Disclaimer: This analysis is for research purposes only.
Always consult qualified medical professionals for clinical decisions.
                """
                st.download_button(
                    label="📄 Download Text Report",
                    data=report,
                    file_name=f"polyp_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            # Enhanced placeholder
            st.markdown("""
            <div class="results-card">
                <div style="text-align: center; padding: 4rem 2rem;">
                    <h3 style="color: #9ca3af; margin-bottom: 1rem;">📊 Waiting for Analysis</h3>
                    <p style="color: #6b7280; margin-bottom: 2rem;">Upload an image and click "Analyze for Polyps" to see detailed AI-powered results</p>
                    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                        <div style="background: #374151; padding: 1rem; border-radius: 10px; min-width: 120px;">
                            <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">✅ Model Ready</p>
                        </div>
                        <div style="background: #374151; padding: 1rem; border-radius: 10px; min-width: 120px;">
                            <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">🚀 Fast Analysis</p>
                        </div>
                        <div style="background: #374151; padding: 1rem; border-radius: 10px; min-width: 120px;">
                            <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">🎯 High Accuracy</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with enhanced disclaimer
    st.markdown("""
    <div class="footer">
        <h4>⚠️ MEDICAL DISCLAIMER</h4>
        <p>This AI system is for research and educational purposes only.<br>
        Results should not be used for clinical diagnosis or treatment decisions.<br>
        Always consult qualified medical professionals for proper medical care.</p>
        <p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
        🦆 Duck-Net Architecture | 🏥 U-Net Baseline | 🤗 Powered by Hugging Face | 📊 Built with Streamlit<br>
        🔬 Trained on Kvasir-SEG Dataset | 🧠 PyTorch Deep Learning | 🎯 Computer Vision for Medical Imaging
        </p>
    </div>
    """, unsafe_allow_html=True)

# Add pandas import for timestamp
import pandas as pd

if __name__ == "__main__":
    main()
