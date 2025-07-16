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

# Page configuration
st.set_page_config(
    page_title="AI Polyp Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling with blue and dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --dark-blue: #1e40af;
        --light-blue: #3b82f6;
        --bg-dark: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border: #475569;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    .main {
        background: var(--bg-dark);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, var(--primary-blue), var(--dark-blue));
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
    }
    
    .app-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    .app-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-secondary);
        width: 320px !important;
    }
    
    .css-1lcbmhc, .css-1cypcdb, .css-17eq0hr {
        width: 320px !important;
        min-width: 320px !important;
    }
    
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        width: 320px !important;
        min-width: 320px !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 320px !important;
        min-width: 320px !important;
        background: var(--bg-secondary);
    }
    
    /* Cards */
    .control-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .control-card h3 {
        color: var(--light-blue);
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .result-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .upload-card {
        background: var(--bg-card);
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: var(--light-blue);
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue), var(--dark-blue)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        height: auto !important;
        min-height: 44px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
        background: linear-gradient(135deg, var(--light-blue), var(--primary-blue)) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    .stButton > button p {
        color: white !important;
        margin: 0 !important;
    }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--light-blue);
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Alerts */
    .alert {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }
    
    .alert-success {
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success);
        color: var(--success);
    }
    
    .alert-error {
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--error);
        color: var(--error);
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border-color: var(--warning);
        color: var(--warning);
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed var(--border);
        border-radius: 12px;
        background: var(--bg-card);
        padding: 2rem;
        text-align: center;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--light-blue);
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--bg-card);
    }
    
    .stSlider > div > div > div > div {
        color: var(--light-blue);
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
    }
    
    /* Example buttons */
    .example-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .example-btn {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem;
        color: var(--light-blue);
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-btn:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: var(--light-blue);
    }
    
    /* Results section */
    .results-empty {
        text-align: center;
        padding: 3rem;
        color: var(--text-muted);
        font-size: 1.1rem;
    }
    
    .results-empty::before {
        content: "🔬";
        display: block;
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .css-1lcbmhc, .css-1cypcdb, .css-17eq0hr, [data-testid="stSidebar"] {
            width: 280px !important;
            min-width: 280px !important;
        }
        
        .app-header h1 {
            font-size: 2rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr 1fr;
        }
    }
    
    @media (max-width: 480px) {
        .css-1lcbmhc, .css-1cypcdb, .css-17eq0hr, [data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
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
        with st.spinner("Loading AI model..."):
            model_path = hf_hub_download(
                repo_id="ibrahim313/unet-adam-diceloss",
                filename="pytorch_model.bin"
            )
            
            model = UNET(ch=32)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            return model, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {e}"

def load_example_image(image_url):
    """Load example image from repository"""
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
        return None, None, None, "Model not loaded!"
    
    if image is None:
        return None, None, None, "Please upload an image first!"
    
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
        
        # Create visualization
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1e293b')
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, color='white', pad=15)
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(pred_mask, cmap='Blues')
        axes[1].set_title('Detection Mask', fontsize=12, color='white', pad=15)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(pred_mask, cmap='Reds', alpha=0.6)
        axes[2].set_title('Detection Overlay', fontsize=12, color='white', pad=15)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1e293b')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image, polyp_percentage, int(polyp_pixels), total_pixels
        
    except Exception as e:
        return None, None, None, f"Error processing image: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>AI Polyp Detection System</h1>
        <p>Advanced Medical Image Analysis Platform</p>
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
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">Model Controls</div>', unsafe_allow_html=True)
        
        # Detection sensitivity
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown('<h3>🎯 Detection Sensitivity</h3>', unsafe_allow_html=True)
        threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Higher values = more sensitive detection"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example images
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown('<h3>📸 Sample Images</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🖼️ Sample 1", use_container_width=True):
                example_img = load_example_image("https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0qoxqj9q6s0835b43399p4.jpg")
                if example_img:
                    st.session_state.example_image = example_img
        
        with col2:
            if st.button("🖼️ Sample 2", use_container_width=True):
                example_img = load_example_image("https://github.com/muhammadibrahim313/polyp_Detection/raw/main/cju0roawvklrq0799vmjorwfv.jpg")
                if example_img:
                    st.session_state.example_image = example_img
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model info
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown('<h3>ℹ️ Model Information</h3>', unsafe_allow_html=True)
        st.markdown("""
        **Architecture:** U-Net  
        **Input Size:** 384×384  
        **Developer:** Asim Khan  
        **Repository:** ibrahim313/unet-adam-diceloss  
        **Status:** ✅ Ready
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="section-header">Image Upload</div>', unsafe_allow_html=True)
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload colonoscopy image",
            type=['jpg', 'jpeg', 'png'],
            help="Select a colonoscopy image for analysis"
        )
        
        # Display image
        display_image = None
        if uploaded_file is not None:
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image", use_container_width=True)
        elif 'example_image' in st.session_state:
            display_image = st.session_state.example_image
            st.image(display_image, caption="Sample Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if display_image is not None:
                with st.spinner("Analyzing image..."):
                    result_image, polyp_percentage, polyp_pixels, total_pixels = predict_polyp(display_image, threshold)
                    
                    if result_image is not None and polyp_percentage is not None:
                        st.session_state.results = {
                            'image': result_image,
                            'percentage': polyp_percentage,
                            'pixels': polyp_pixels,
                            'total_pixels': total_pixels,
                            'threshold': threshold
                        }
                        st.success("Analysis completed!")
                    else:
                        st.error(f"Analysis failed: {total_pixels}")
            else:
                st.warning("Please upload an image first!")
    
    with col2:
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display result image
            if results['image']:
                st.image(results['image'], use_container_width=True)
            
            # Metrics
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <p class="metric-value">{results['percentage']:.2f}%</p>
                <p class="metric-label">Coverage</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <p class="metric-value">{results['pixels']:,}</p>
                <p class="metric-label">Detected Pixels</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <p class="metric-value">{results['threshold']}</p>
                <p class="metric-label">Threshold</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical assessment
            if results['pixels'] > 100:
                st.markdown("""
                <div class="alert alert-error">
                    <strong>⚠️ POLYP DETECTED</strong><br>
                    Medical review recommended. Please consult a healthcare professional.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert alert-success">
                    <strong>✅ NO POLYP DETECTED</strong><br>
                    No significant polyp features found in this image.
                </div>
                """, unsafe_allow_html=True)
            
            # Technical details
            with st.expander("Technical Details"):
                st.markdown(f"""
                **Detection Threshold:** {results['threshold']}  
                **Total Pixels:** {results['total_pixels']:,}  
                **Detected Pixels:** {results['pixels']:,}  
                **Processing Time:** < 1 second  
                """)
        
        else:
            st.markdown("""
            <div class="results-empty">
                Upload an image and click analyze to see results
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
