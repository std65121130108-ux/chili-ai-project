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

# --- 2. 🎨 CSS ตกแต่ง (Theme แดง-ชมพู ตามที่คุณขอ) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* บังคับฟอนต์ Prompt ทั้งหน้า */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* 1. พื้นหลังหลัก (Background): Gradient แดง-ชมพู */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. Animation Keyframes */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    /* 3. ปรับแต่ง "กรอบ/การ์ด" (Glass Card สีขาว) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.95) !important; /* สีขาวโปร่งแสง */
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
        
        /* จัดตำแหน่งและขนาด */
        max-width: 480px !important;
        width: 100% !important;
        margin: 0 auto 20px auto !important;
        padding: 40px 30px !important;
        
        /* Animation */
        animation: fadeUp 0.8s ease-out !important;
    }
    
    /* ซ่อน Header/Footer เดิม */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 4. จัดการ Typography */
    .icon-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .app-icon {
        font-size: 4rem;
        background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        animation: pulse 2s infinite;
    }
    h1 {
        color: #333 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin: 0 0 5px 0 !important;
        text-align: center !important;
    }
    .subtitle {
        color: #d32f2f;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 5px;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .description {
        color: #666;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 25px;
    }

    /* 5. ปุ่มกด (Button Styling) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
    }
    
    /* 6. File Uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(249, 249, 249, 0.8) !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    /* Footer Credit */
    .footer-credit {
        font-size: 0.8rem;
        color: #999;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        # ใส่โค้ดดาวน์โหลด Model ของคุณตรงนี้ถ้าจำเป็น
        pass
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล (UI) ---

model = load_model()

# สร้าง Container (Card สีขาว)
with st.container(border=True):
    
    # Header ส่วนบน (เลียนแบบ HTML ที่ให้มา)
    st.markdown("""
        <div class="icon-container">
            <div class="app-icon">🌶️</div>
        </div>
        <div class="subtitle">AI Expert System</div>
        <h1>Chili Doctor AI</h1>
        <p class="description">
            ระบบวินิจฉัยโรคพริกอัจฉริยะ ด้วยเทคโนโลยี<br>
            <strong>Deep Learning (EfficientNetB4)</strong>
        </p>
    """, unsafe_allow_html=True)

    # พื้นที่อัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        
        # แสดงรูปภาพ
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 10, 1]) # ปรับให้รูปกว้างขึ้นเต็มการ์ด
        with col2:
            st.image(image, use_container_width=True)
        
        # ปุ่ม Analyze
        if st.button("🚀 วินิจฉัยโรค"):
            if model is None:
                st.error("❌ ไม่พบไฟล์โมเดล")
            else:
                with st.spinner('กำลังประมวลผล...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr style='margin: 25px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                # แสดงผลลัพธ์
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="color: #888; font-size: 0.9rem;">ผลการวิเคราะห์</div>
                        <h2 style="color: #FF4B2B; margin: 10px 0;">{result_class.upper()}</h2>
                        <span style="background: #fff0f0; color: #FF4B2B; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                            ความแม่นยำ: {confidence:.2f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                # คำแนะนำ (Treatment)
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                
                if result_class == 'healthy':
                    treatment_text = "🌿 <b>ต้นพริกแข็งแรงดี!</b><br>ไม่พบร่องรอยของโรค ดูแลรดน้ำตามปกติ"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                elif result_class == 'leaf curl':
                    treatment_text = "🍂 <b>โรคใบหงิก:</b><br>ระวังแมลงพาหะ กำจัดวัชพืช และใช้น้ำหมักชีวภาพ"
                elif result_class == 'leaf spot':
                    treatment_text = "🌑 <b>โรคใบจุดตากบ:</b><br>ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
                elif result_class == 'whitefly':
                    treatment_text = "🪰 <b>แมลงหวี่ขาว:</b><br>ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                elif result_class == 'yellow':
                    treatment_text = "🟡 <b>อาการใบเหลือง:</b><br>อาจขาดธาตุอาหาร ให้ตรวจสอบสภาพดินและใส่ปุ๋ยบำรุง"
                
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 16px; margin-top: 20px; font-size: 0.95rem; text-align: left; line-height: 1.6;">
                        {treatment_text}
                    </div>
                """, unsafe_allow_html=True)

    # Footer ในการ์ด
    st.markdown("""
        <div class="footer-credit">
            โครงงานวิจัยทางคอมพิวเตอร์ • <strong>UBRU</strong><br>
            พัฒนาโดย: WhiteCat Team และผองเพื่อน
        </div>
    """, unsafe_allow_html=True)