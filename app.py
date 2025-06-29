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
import base64
from io import BytesIO

# Page configuration with dark theme
st.set_page_config(
    page_title="🏥 AI Polyp Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #0c0c0c 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0;
        font-weight: 400;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left: 4px solid #0ea5e9;
    }
    
    .info-card h3 {
        color: #0ea5e9;
        margin-top: 0;
        font-weight: 600;
    }
    
    .info-card p {
        color: #b8b8b8;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Results card */
    .results-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Success/Error alerts */
    .success-alert {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #0c0c0c 100%);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0ea5e9;
        margin: 0;
    }
    
    .metric-label {
        color: #b8b8b8;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #1a1a1a 0%, #0c0c0c 100%);
        border-top: 2px solid #333;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border-radius: 10px;
    }
    
    .footer h4 {
        color: #dc2626;
        margin: 0;
        font-weight: 600;
    }
    
    .footer p {
        color: #9ca3af;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Your UNET Model Definition
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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Global variables
device = torch.device('cpu')
transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
    ToTensorV2()
])

@st.cache_resource
def load_model():
    """Load model from HuggingFace repository"""
    try:
        with st.spinner("🔄 Loading AI model from HuggingFace..."):
            model_path = hf_hub_download(
                repo_id="ibrahim313/unet-adam-diceloss",
                filename="pytorch_model.bin"
            )
            
            model = UNET(ch=32)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            return model, "✅ Model loaded successfully!"
    except Exception as e:
        return None, f"❌ Error loading model: {e}"

def load_example_image(image_url):
    """Load example image from GitHub repository"""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            st.error(f"Failed to load example image. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error loading example image: {e}")
        return None

def predict_polyp(image, threshold=0.5):
    """Predict polyp in uploaded image"""
    if st.session_state.model is None:
        return None, "❌ Model not loaded! Please wait for model to load.", None, None
    
    if image is None:
        return None, "❌ Please upload an image first!", None, None
    
    try:
        # Convert image to numpy array
        if isinstance(image, Image.Image):
            original_image = np.array(image.convert('RGB'))
        else:
            original_image = np.array(image)
        
        # Preprocess image
        transformed = transform(image=original_image)
        input_tensor = transformed['image'].unsqueeze(0).float()
        
        # Make prediction
        with torch.no_grad():
            prediction = st.session_state.model(input_tensor)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > threshold).float()
        
        # Convert to numpy
        pred_mask = prediction.squeeze().cpu().numpy()
        
        # Calculate metrics
        polyp_pixels = np.sum(pred_mask)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        polyp_percentage = (polyp_pixels / total_pixels) * 100
        
        # Create visualization with dark theme
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('🖼️ Original Image', fontsize=14, color='white', pad=20)
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('🎭 Predicted Mask', fontsize=14, color='white', pad=20)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(pred_mask, cmap='Reds', alpha=0.6)
        axes[2].set_title('🔍 Detection Overlay', fontsize=14, color='white', pad=20)
        axes[2].axis('off')
        
        # Add main title with results
        if polyp_pixels > 100:
            main_title = f"🚨 POLYP DETECTED! Coverage: {polyp_percentage:.2f}%"
            title_color = '#ef4444'
        else:
            main_title = f"✅ No Polyp Detected - Coverage: {polyp_percentage:.2f}%"
            title_color = '#10b981'
        
        fig.suptitle(main_title, fontsize=16, fontweight='bold', color=title_color, y=0.95)
        plt.tight_layout()
        
        # Save plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image, polyp_percentage, polyp_pixels, total_pixels
        
    except Exception as e:
        return None, None, None, f"❌ Error processing image: {str(e)}"

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Polyp Detection System</h1>
        <p>Advanced Medical Imaging with Deep Learning</p>
        <p style="opacity: 0.9;">Upload colonoscopy images for intelligent polyp detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        model, status = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success(status)
        else:
            st.error(status)
            return
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("### 🔧 Controls")
        
        # Model info
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Model Information</h3>
            <p><strong>Repository:</strong> ibrahim313/unet-adam-diceloss</p>
            <p><strong>Architecture:</strong> U-Net (32 channels)</p>
            <p><strong>Dataset:</strong> Kvasir-SEG (1000 images)</p>
            <p><strong>Status:</strong> ✅ Ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detection sensitivity
        threshold = st.slider(
            "🎯 Detection Sensitivity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Higher values = more sensitive detection"
        )
        
        st.markdown("### 📸 Example Images")
        
        # Example images
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🖼️ Example 1", use_container_width=True):
                example_img = load_example_image("https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0qoxqj9q6s0835b43399p4.jpg")
                if example_img:
                    st.session_state.example_image = example_img
        
        with col2:
            if st.button("🖼️ Example 2", use_container_width=True):
                example_img = load_example_image("https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0roawvklrq0799vmjorwfv.jpg")
                if example_img:
                    st.session_state.example_image = example_img
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a colonoscopy image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a colonoscopy image for polyp detection"
        )
        
        # Display image
        display_image = None
        if uploaded_file is not None:
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image", use_column_width=True)
        elif 'example_image' in st.session_state:
            display_image = st.session_state.example_image
            st.image(display_image, caption="Example Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze for Polyps", type="primary", use_container_width=True):
            if display_image is not None:
                with st.spinner("🔄 Analyzing image..."):
                    result_image, polyp_percentage, polyp_pixels, error = predict_polyp(display_image, threshold)
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state.results = {
                            'image': result_image,
                            'percentage': polyp_percentage,
                            'pixels': polyp_pixels,
                            'threshold': threshold
                        }
            else:
                st.warning("Please upload an image or select an example first!")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display result image
            if results['image']:
                st.image(results['image'], use_column_width=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{results['percentage']:.3f}%</p>
                    <p class="metric-label">Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{int(results['pixels']):,}</p>
                    <p class="metric-label">Detected Pixels</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{results['threshold']}</p>
                    <p class="metric-label">Threshold</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clinical assessment
            if results['pixels'] > 100:
                st.markdown("""
                <div class="error-alert">
                    <h4>🚨 POLYP DETECTED</h4>
                    <p><strong>Recommendation:</strong> Medical review recommended</p>
                    <p>A potential polyp has been identified in the image. Please consult with a qualified medical professional for proper diagnosis and treatment planning.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-alert">
                    <h4>✅ NO POLYP DETECTED</h4>
                    <p><strong>Recommendation:</strong> Continue routine monitoring</p>
                    <p>No significant polyp features detected in this image. Continue with regular screening as recommended by your healthcare provider.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Technical details
            with st.expander("🔬 Technical Details"):
                st.markdown(f"""
                **Model Architecture:** U-Net with skip connections  
                **Input Size:** 384×384 pixels  
                **Base Channels:** 32  
                **Detection Threshold:** {results['threshold']}  
                **Total Pixels:** {384*384:,}  
                **Processing Time:** < 1 second  
                """)
        
        else:
            st.markdown("""
            <div class="results-card">
                <p style="text-align: center; color: #9ca3af; font-size: 1.1rem;">
                    📸 Upload an image and click "Analyze for Polyps" to see detailed results
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with disclaimer
    st.markdown("""
    <div class="footer">
        <h4>⚠️ MEDICAL DISCLAIMER</h4>
        <p>This AI system is for research and educational purposes only.<br>
        Always consult qualified medical professionals for clinical decisions.</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">
        🔬 Powered by PyTorch | 🤗 Hosted on Hugging Face | 📊 Streamlit Interface
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
