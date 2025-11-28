import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Chili Doctor AI",
    page_icon="🌶️",
    layout="centered"
)

# --- 2. 🎨 CSS ตกแต่ง (Glassmorphism Card 480px) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก (Background): Gradient สีส้มแดง */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. Animation Keyframes (fadeUp) */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 3. ปรับแต่ง "กรอบ/การ์ด" (Container) ตาม CSS ที่คุณให้มา */
    [data-testid="stVerticalBlockBorderWrapper"] {
        /* CSS จากที่คุณส่งมา */
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
        
        /* จัดตำแหน่ง */
        max-width: 480px !important;
        width: 100% !important;
        margin: 0 auto 20px auto !important; /* จัดกึ่งกลาง */
        padding: 40px 20px !important;
        overflow: hidden !important;
        text-align: center !important;
        
        /* Animation */
        animation: fadeUp 0.8s ease-out !important;
        transform: translateY(0);
        transition: transform 0.3s ease;
    }
    
    /* ซ่อน Header/Footer เดิมของ Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 4. จัดการข้อความ (Typography) ให้ตรงตามแบบ */
    .emoji-icon {
        font-size: 60px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .main-title {
        color: #333;
        font-weight: 700;
        font-size: 1.8rem; /* ประมาณ 28-30px */
        margin: 0;
        line-height: 1.2;
    }
    .sub-title {
        color: #555;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 5px;
    }
    .tech-tag {
        color: #888;
        font-size: 0.8rem;
        margin-top: 5px;
        margin-bottom: 20px;
        font-weight: 300;
    }

    /* 5. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
    }
    
    /* 6. File Uploader Styling */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #fafafa !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    /* ปรับแต่งข้อความใน Dropzone */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Drag and drop file here";
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        display: block;
        margin-bottom: 5px;
    }
    [data-testid="stFileUploaderDropzone"] div div small {
        font-size: 0.8rem;
        color: #999;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: rgba(255,255,255,0.8);
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        file_id = '1tURhAR8mXLAgnuU3EULswpcFGxnalWAV'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.status("⏳ กำลังดาวน์โหลดโมเดล...", expanded=True) as status:
            try:
                import gdown
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    status.update(label="✅ เสร็จสิ้น!", state="complete", expanded=False)
                else:
                    return None
            except:
                return None
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

# ฟังก์ชันทำนาย
def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล (UI) ---

model = load_model()

# สร้าง Container (Card สีขาว) ด้วย CSS .glass-card
with st.container(border=True):
    
    # ส่วนหัว (Header) - จัดข้อความตามที่คุณต้องการ
    st.markdown("""
        <div style="text-align: center;">
            <div class="emoji-icon">🌶️</div>
            <div class="main-title">Chili Doctor AI</div>
            <div class="sub-title">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <div class="tech-tag">Deep Learning Technology (EfficientNetB4)</div>
        </div>
    """, unsafe_allow_html=True)

    # ส่วนอัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        
        # แสดงชื่อไฟล์และขนาด (จำลอง UI)
        file_details = {"FileName":file.name, "FileType":file.type,"FileSize":file.size}
        size_kb = file.size / 1024
        
        # แสดงรูปภาพ
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        # แสดงชื่อไฟล์ (ตามภาพตัวอย่าง)
        st.markdown(f"""
            <div style="text-align: left; margin-top: 10px; font-size: 0.85rem; color: #555; word-wrap: break-word;">
                <strong>File:</strong> {file.name}<br>
                <span style="color: #999;">{size_kb:.1f}KB</span>
            </div>
        """, unsafe_allow_html=True)
            
        if st.button("🔍 Analyze Image"):
            if model is None:
                st.error("❌ Model Error")
            else:
                with st.spinner('Processing...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                # ผลลัพธ์
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="color: #888; font-size: 0.9rem;">Result</div>
                        <h2 style="color: #FF4B2B; margin: 5px 0;">{result_class}</h2>
                        <div style="background: #eee; padding: 4px 12px; border-radius: 12px; display: inline-block; font-size: 0.8rem; color: #555;">
                            Confidence: {confidence:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # คำแนะนำ
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                
                if result_class == 'healthy':
                    treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                elif result_class == 'leaf curl':
                    treatment_text = "โรคใบหงิก: กำจัดวัชพืชและใช้สารสกัดสะเดา"
                elif result_class == 'leaf spot':
                    treatment_text = "โรคใบจุดตากบ: ตัดแต่งใบที่เป็นโรคและฉีดพ่นสารป้องกันเชื้อรา"
                elif result_class == 'whitefly':
                     treatment_text = "แมลงหวี่ขาว: ใช้กับดักกาวเหนียวหรือน้ำหมักสมุนไพร"
                elif result_class == 'yellow':
                     treatment_text = "ใบเหลือง: ตรวจสอบสภาพดินและใส่ปุ๋ยบำรุง"
                
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color}; padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 0.9rem; text-align: left;">
                        {treatment_text}
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        Designed by WhiteCat Team
    </div>
""", unsafe_allow_html=True)