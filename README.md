# 🏥 Polyp Detection Streamlit App - Setup Guide

## 📋 Requirements.txt
```txt
streamlit>=1.28.0
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
Pillow>=9.0.0
matplotlib>=3.5.0
albumentations>=1.3.0
huggingface-hub>=0.16.0
requests>=2.28.0
```

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the App
- Open your browser and go to `http://localhost:8501`
- The app will automatically load the AI model from HuggingFace

## 🎯 Features

### **Professional Dark UI**
- Modern gradient dark theme
- Medical-grade professional styling
- Responsive design for all screen sizes
- Interactive hover effects and animations

### **AI Model Integration**
- Loads U-Net model from `ibrahim313/unet-adam-diceloss`
- Real-time polyp detection and segmentation
- Adjustable sensitivity threshold
- Detailed analysis with metrics

### **Example Images**
- Two test colonoscopy images from your GitHub repo
- One-click loading for quick testing
- Direct integration with your repository

### **Medical Dashboard**
- Professional metrics display
- Clinical assessment recommendations
- Technical details and model information
- Medical disclaimer and safety warnings

## 📁 File Structure
```
polyp_detection_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Setup and usage guide
└── .streamlit/           # Streamlit config (optional)
    └── config.toml       # App configuration
```

## 🔧 Optional: Streamlit Configuration

Create `.streamlit/config.toml` for custom settings:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0c0c0c"
secondaryBackgroundColor = "#1a1a1a"
textColor = "#ffffff"

[server]
maxUploadSize = 200
enableCORS = false
```

## 🌐 Deployment Options

### **Streamlit Cloud**
1. Push code to GitHub repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from repository

### **Local Network**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## 🎨 Customization

### **Color Scheme**
- Primary: `#667eea` (Blue gradient)
- Secondary: `#764ba2` (Purple gradient)
- Success: `#10b981` (Green)
- Error: `#ef4444` (Red)
- Warning: `#f59e0b` (Orange)

### **Typography**
- Font Family: Inter (Google Fonts)
- Professional medical app styling
- Consistent spacing and hierarchy

## 🔐 Security Notes

- Model loads from trusted HuggingFace repository
- Example images from your verified GitHub repo
- No data persistence or user tracking
- Medical disclaimer included for safety

## 📞 Support

For issues or customizations:
- Check Streamlit documentation
- Verify HuggingFace model access
- Ensure all dependencies are installed
- Test with example images first
