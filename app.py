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

# --- 2. 🎨 CSS ตกแต่ง (Design: Clean White Card) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* บังคับฟอนต์ Prompt */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* 1. Background: Gradient เต็มจอ */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. Main White Card (กรอบสีขาวหลัก) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important; /* สีขาวทึบ */
        border-radius: 30px !important; /* ขอบมนมาก */
        border: none !important; /* ไม่เอาเส้นขอบสีเทา */
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15) !important; /* เงานุ่มๆ ฟุ้งๆ */
        padding: 40px 30px !important;
        max-width: 550px;
        margin: auto;
    }

    /* 3. Typography: ปรับสีตัวหนังสือให้เข้มขึ้น เพราะอยู่บนพื้นขาว */
    h1 {
        color: #FF4B2B !important; /* หัวข้อสีแดง */
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 5px !important;
        text-align: center;
    }
    
    .subtitle {
        color: #666 !important;
        font-size: 1rem !important;
        font-weight: 400;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .tech-badge {
        background: #ffebee;
        color: #c62828;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* 4. Upload Area */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8f9fa !important; /* สีเทาอ่อนๆ */
        border: 2px dashed #FF4B2B !important; /* เส้นประสีแดง */
        border-radius: 20px !important;
        padding: 30px 20px !important;
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Drag & Drop Image Here";
        color: #555;
        font-weight: 600;
    }

    /* 5. Button */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3) !important;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 20px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px rgba(255, 75, 43, 0.5) !important;
    }
    div.stButton > button p {
        color: white !important;
    }

    /* Result Section Styling */
    .result-header {
        text-align: center;
        margin-top: 30px;
        border-top: 1px solid #eee;
        padding-top: 30px;
    }
    .result-title {
        color: #FF4B2B;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .confidence-badge {
        background: #FF4B2B;
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(255, 75, 43, 0.3);
    }
    .recommendation-box {
        background-color: #f8f9fa;
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
        display: flex;
        align-items: start;
        border-left: 5px solid #FF4B2B;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
    }

    #MainMenu, header, footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
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

# --- 4. ส่วนแสดงผล UI ---

model = load_model()

# สร้าง Container (กรอบสีขาว)
with st.container(border=True):
    
    # Header
    st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 10px;">🌶️</div>
            <h1>Chili Doctor AI</h1>
            <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <span class="tech-badge">Deep Learning (EfficientNetB4)</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # พื้นที่อัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        
        # แสดงรูปภาพ
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # ปุ่มกด
        if st.button("🚀 เริ่มการวินิจฉัย (Start Diagnosis)"):
            if model is None:
                st.error("⚠️ Model file not found.")
            else:
                with st.spinner('กำลังวิเคราะห์...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # --- ส่วนแสดงผลลัพธ์ (จัดให้อยู่ในกรอบขาวเดียวกัน) ---
                
                # เตรียมข้อมูลคำแนะนำ
                treatment_text = ""
                icon = ""
                
                if result_class == 'Healthy':
                    treatment_text = "ต้นพริกแข็งแรงดีมาก! แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยบำรุงตามปกติ"
                    icon = "🌿"
                elif result_class == 'Leaf Curl':
                    treatment_text = "โรคใบหงิก: ระวังแมลงพาหะ (เช่น แมลงหวี่ขาว) กำจัดวัชพืช และใช้สารสกัดสะเดาฉีดพ่น"
                    icon = "🍂"
                elif result_class == 'Leaf Spot':
                    treatment_text = "โรคใบจุด: เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันกำจัดเชื้อรา"
                    icon = "🌑"
                elif result_class == 'Whitefly':
                    treatment_text = "แมลงหวี่ขาว: เป็นพาหะนำโรค ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพรไล่แมลง"
                    icon = "🪰"
                elif result_class == 'Yellow':
                    treatment_text = "อาการใบเหลือง: อาจเกิดจากการขาดธาตุอาหาร ตรวจสอบสภาพดินและใส่ปุ๋ยบำรุง"
                    icon = "🟡"

                # แสดงผล (ใช้ HTML Class ที่เขียนไว้ใน CSS ด้านบน)
                st.markdown(f"""
                    <div class="result-header">
                        <div style="color: #999; font-size: 0.9rem;">ผลการวิเคราะห์</div>
                        <div class="result-title">{result_class.upper()}</div>
                        <div class="confidence-badge">
                            ความแม่นยำ: {confidence:.2f}%
                        </div>
                    </div>
                    
                    <div class="recommendation-box">
                        <div style="font-size: 2rem; margin-right: 20px;">{icon}</div>
                        <div>
                            <h4 style="margin: 0 0 5px 0; color: #333;">คำแนะนำการดูแล</h4>
                            <p style="color: #555; line-height: 1.6; margin: 0;">{treatment_text}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer นอกกรอบขาว
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        Designed by WhiteCat Team
    </div>
""", unsafe_allow_html=True)