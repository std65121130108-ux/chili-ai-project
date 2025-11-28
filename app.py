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

# --- 2. 🎨 CSS ตกแต่ง (Design: Clean & Borderless) ---
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

    /* 2. Typography: ปรับให้เป็นสีขาว เพื่อให้เด่นบนพื้นแดง */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        text-shadow: 0 4px 10px rgba(0,0,0,0.2); /* เงาตัวหนังสือ */
        margin-bottom: 0 !important;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        font-weight: 300;
        margin-bottom: 20px;
    }
    
    .tech-badge {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* 3. Upload Area: ปรับให้ดูเป็นกระจกฝ้าจางๆ บนพื้นแดง */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 2px dashed rgba(255, 255, 255, 0.6) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(255, 255, 255, 0.25) !important;
        border-color: #fff !important;
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Drag & Drop Image Here";
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: rgba(255,255,255,0.8) !important;
    }

    /* 4. Button: ปุ่มสีขาว ตัวหนังสือสีแดง (Reverse Style) */
    div.stButton > button {
        background: white !important;
        color: #FF4B2B !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15) !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px rgba(0,0,0,0.2) !important;
    }
    div.stButton > button p {
        color: #FF4B2B !important;
    }

    /* 5. Result Card: กล่องสีขาวที่จะโผล่มาตอนแสดงผล */
    .result-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        margin-top: 30px;
        color: #333;
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .main-icon {
        font-size: 5rem;
        display: inline-block;
        animation: bounce 3s infinite;
        filter: drop-shadow(0 10px 10px rgba(0,0,0,0.2));
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: rgba(255,255,255,0.6);
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

# Header ส่วนบน (อยู่นอก Container แล้ว)
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div class="main-icon">🌶️</div>
        <h1>Chili Doctor AI</h1>
        <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
        <span class="tech-badge">Powered by EfficientNetB4</span>
    </div>
""", unsafe_allow_html=True)

# พื้นที่อัปโหลด
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    
    # แสดงรูปภาพ (ใส่กรอบขาวบางๆ ให้รูปเด่นขึ้น)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 20px;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # รายละเอียดไฟล์
    size_kb = file.size / 1024
    st.markdown(f"""
        <div style="text-align: center; margin-top: 15px; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
            📷 {file.name} • {size_kb:.1f} KB
        </div>
    """, unsafe_allow_html=True)
        
    # ปุ่มกด (จะอยู่ด้านล่างรูป)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 วินิจฉัยโรค (START)"):
        if model is None:
            st.error("⚠️ Model file not found.")
        else:
            with st.spinner('กำลังวิเคราะห์...'):
                predictions = import_and_predict(image, model)
                class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100

            # --- ส่วนแสดงผลลัพธ์ (ใช้ Card สีขาวเพื่อให้ข้อความอ่านง่าย) ---
            
            # เตรียมข้อมูลคำแนะนำ
            treatment_title = "คำแนะนำการดูแล"
            treatment_text = ""
            bg_icon_color = "#eee"
            icon = "🔬"
            diagram_tag = "" # Placeholder for diagram tag

            if result_class == 'Healthy':
                treatment_text = "ต้นพริกแข็งแรงดีมาก! แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยบำรุงตามปกติ"
                bg_icon_color = "#d4edda" 
                icon = "🌿"
            elif result_class == 'Leaf Curl':
                treatment_text = "โรคใบหงิกมักเกิดจากแมลงหวี่ขาวหรือเพลี้ยไฟเป็นพาหะ แนะนำให้กำจัดวัชพืช และใช้สารสกัดสะเดาฉีดพ่น"
                bg_icon_color = "#fff3cd"
                icon = "🍂"
                diagram_tag = ""
            elif result_class == 'Leaf Spot':
                treatment_text = "โรคใบจุดเกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันกำจัดเชื้อรา"
                bg_icon_color = "#ffcdd2"
                icon = "🌑"
                diagram_tag = ""
            elif result_class == 'Whitefly':
                treatment_text = "แมลงหวี่ขาวเป็นพาหะนำโรคสำคัญ ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                bg_icon_color = "#e1f5fe"
                icon = "🪰"
                diagram_tag = ""
            elif result_class == 'Yellow':
                treatment_text = "อาการใบเหลืองอาจเกิดจากการขาดธาตุอาหาร (เช่น ไนโตรเจน) ควรตรวจสอบสภาพดินและใส่ปุ๋ยบำรุง"
                bg_icon_color = "#fff9c4"
                icon = "🟡"
                diagram_tag = ""

            # แสดงผลในกล่องสีขาว (Result Card)
            st.markdown(f"""
                <div class="result-card">
                    <div style="text-align: center; border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 20px;">
                        <div style="color: #888; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">ผลการวิเคราะห์</div>
                        <h2 style="color: #FF4B2B; font-size: 2.2rem; margin: 10px 0; font-weight: 700;">{result_class.upper()}</h2>
                        <span style="background: #FF4B2B; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
                            ความแม่นยำ: {confidence:.2f}%
                        </span>
                    </div>
                    
                    <div style="display: flex; align-items: start;">
                        <div style="background: {bg_icon_color}; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin-right: 15px; flex-shrink: 0;">
                            {icon}
                        </div>
                        <div>
                            <h4 style="margin: 0 0 5px 0; color: #333;">{treatment_title}</h4>
                            <p style="color: #555; line-height: 1.6; margin: 0;">{treatment_text}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Trigger diagram for educational purpose if a disease is found
            if result_class != 'Healthy':
                 st.markdown(f"", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        Designed by WhiteCat Team
    </div>
""", unsafe_allow_html=True)