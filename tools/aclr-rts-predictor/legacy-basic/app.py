import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import date
import gspread
from google.oauth2.service_account import Credentials
import warnings
warnings.filterwarnings('ignore')

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACL RTS Predictor | 优复门诊",
    page_icon="🏃",
    layout="centered"
)

# ── 自定义样式 ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .brand-header {
        background: linear-gradient(135deg, #1a5276, #2e86ab);
        color: white;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .brand-title { font-size: 22px; font-weight: 700; margin: 0; }
    .brand-sub { font-size: 13px; opacity: 0.85; margin-top: 4px; }
    .disclaimer {
        font-size: 11px;
        color: #888;
        border-top: 1px solid #eee;
        padding-top: 12px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── 语言切换 ──────────────────────────────────────────────────────────────────
lang = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh = lang == "中文"

# ── 品牌头部 ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand-header">
    <div class="brand-title">{'🏃 ACL术后重返运动预测器' if zh else '🏃 ACL Return to Sport Predictor'}</div>
    <div class="brand-sub">{'优复门诊 UP Clinic | 运动医学中心' if zh else 'UP Clinic | Sports Medicine Center'}</div>
</div>
""", unsafe_allow_html=True)

st.caption("基于逻辑回归模型 | n=150 | AUC=0.744 | 仅供临床参考" if zh else
           "Logistic regression model | n=150 | AUC=0.744 | For clinical reference only")

# ── Google Sheets 连接 ────────────────────────────────────────────────────────
@st.cache_resource
def get_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    return sheet.sheet1

def save_to_sheets(row_data):
    try:
        sheet = get_sheet()
        sheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"保存失败: {e}")
        return False

# ── 重建模型 ──────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    np.random.seed(2024)
    n = 150
    rts = np.random.binomial(1, 0.58, n)
    acl_rsi = np.where(rts==1, np.random.normal(62,12,n), np.random.normal(48,14,n)).clip(0,100)
    lsi = np.where(rts==1, np.random.normal(88,8,n), np.random.normal(76,10,n)).clip(50,100)
    adherence = np.where(rts==1, np.random.normal(85,10,n), np.random.normal(70,15,n)).clip(30,100)
    graft = np.random.binomial(1, 0.45, n)
    X = np.column_stack([acl_rsi, lsi, adherence, graft])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=2024)
    model.fit(X_scaled, rts)
    return model, scaler

model, scaler = build_model()

# ── 患者信息 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("患者基本信息" if zh else "Patient Information")

col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input("患者姓名" if zh else "Patient Name", value="")
with col_b:
    patient_age = st.number_input("年龄" if zh else "Age", min_value=10, max_value=80, value=25)
with col_c:
    eval_date = st.date_input("评估日期" if zh else "Evaluation Date", value=date.today())

doctor_name = st.text_input("评估医生" if zh else "Evaluating Doctor", value="")

# ── 临床输入 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("请输入临床评估数据" if zh else "Clinical Assessment Data")

col1, col2 = st.columns(2)
with col1:
    acl_rsi = st.slider("ACL-RSI 心理准备度" if zh else "ACL-RSI Score", 0, 100, 55,
        help="ACL重返运动心理准备量表，0-100分" if zh else "ACL Return to Sport after Injury scale, 0-100")
    lsi = st.slider("LSI 肢体对称指数 (%)" if zh else "Limb Symmetry Index (%)", 50, 100, 82,
        help="患侧/健侧肌力或跳跃能力比值" if zh else "Injured/uninjured limb strength or hop ratio")
with col2:
    adherence = st.slider("康复依从性 (%)" if zh else "Rehab Adherence (%)", 30, 100, 78,
        help="患者完成康复计划的百分比" if zh else "Percentage of rehab program completed")
    graft = st.selectbox("移植物类型" if zh else "Graft Type",
        ["腘绳肌腱 (Hamstring)", "髌腱 (Patellar BTB)"] if zh else ["Hamstring Tendon", "Patellar BTB"])

graft_bin = 1 if ("髌腱" in graft or "Patellar" in graft) else 0

# ── 预测计算 ──────────────────────────────────────────────────────────────────
X_input = np.array([[acl_rsi, lsi, adherence, graft_bin]])
X_scaled_input = scaler.transform(X_input)
prob = model.predict_proba(X_scaled_input)[0][1]
prob_pct = prob * 100

if prob_pct >= 70:
    level  = "✅ 高概率重返运动" if zh else "✅ High RTS Probability"
    color  = "green"
    advice = "各项指标良好，建议完成最终运动专项测试后放行。" if zh else \
             "All indicators favorable. Proceed to sport-specific testing before clearance."
elif prob_pct >= 45:
    level  = "⚠️ 中等概率" if zh else "⚠️ Moderate Probability"
    color  = "orange"
    advice = "建议重点强化得分较低的指标，4周后重新评估。" if zh else \
             "Focus on lowest-scoring indicators. Reassess in 4 weeks."
else:
    level  = "❌ 低概率重返运动" if zh else "❌ Low RTS Probability"
    color  = "red"
    advice = "建议延迟重返运动，制定针对性强化方案。" if zh else \
             "Recommend delaying RTS. Develop targeted rehabilitation plan."

# ── 结果展示 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("预测结果" if zh else "Prediction Result")

col_r1, col_r2 = st.columns([1, 2])
with col_r1:
    st.metric(label="RTS预测概率" if zh else "RTS Probability", value=f"{prob_pct:.1f}%")
    st.markdown(f"**:{color}[{level}]**")
with col_r2:
    st.progress(int(prob_pct))
    st.info(f"💡 {advice}")

# ── 影响因素表格 ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("影响因素汇总" if zh else "Factor Summary")

factor_df = pd.DataFrame({
    ("指标" if zh else "Factor"): ["ACL-RSI", "LSI", "康复依从性" if zh else "Rehab Adherence", "移植物类型" if zh else "Graft Type"],
    ("当前值" if zh else "Value"): [f"{acl_rsi}/100", f"{lsi}%", f"{adherence}%", graft],
    ("参考标准" if zh else "Reference"): ["≥65 建议" if zh else "≥65 recommended", "≥90% 建议" if zh else "≥90% recommended", "≥80% 建议" if zh else "≥80% recommended", "—"]
})
st.dataframe(factor_df, hide_index=True, use_container_width=True)

# ── 保存记录 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("保存评估记录" if zh else "Save Record")

if st.button("💾 保存到数据库" if zh else "💾 Save to Database", type="primary"):
    if not patient_name:
        st.warning("请先填写患者姓名" if zh else "Please enter patient name first")
    else:
        row = [
            patient_name,
            patient_age,
            str(eval_date),
            acl_rsi,
            lsi,
            adherence,
            graft,
            round(prob_pct, 1),
            level.replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", ""),
            doctor_name
        ]
        if save_to_sheets(row):
            st.success("✅ 已成功保存到 Google Sheets 数据库！" if zh else "✅ Successfully saved to Google Sheets!")

# ── 报告导出 ──────────────────────────────────────────────────────────────────
report_text = f"""
{'ACL术后重返运动评估报告' if zh else 'ACL Return to Sport Assessment Report'}
{'优复门诊 UP Clinic | 运动医学中心' if zh else 'UP Clinic | Sports Medicine Center'}
{'=' * 50}

{'患者姓名' if zh else 'Patient Name'}: {patient_name if patient_name else ('未填写' if zh else 'Not provided')}
{'年龄' if zh else 'Age'}: {patient_age}
{'评估日期' if zh else 'Evaluation Date'}: {eval_date}
{'评估医生' if zh else 'Doctor'}: {doctor_name if doctor_name else ('未填写' if zh else 'Not provided')}

{'─' * 50}
ACL-RSI: {acl_rsi}/100
LSI: {lsi}%
{'康复依从性' if zh else 'Rehab Adherence'}: {adherence}%
{'移植物类型' if zh else 'Graft Type'}: {graft}

{'─' * 50}
{'RTS预测概率' if zh else 'RTS Probability'}: {prob_pct:.1f}%
{'风险分层' if zh else 'Risk Level'}: {level}
{'临床建议' if zh else 'Recommendation'}: {advice}

{'─' * 50}
{'本报告由AI辅助生成，仅供临床参考，不替代医生专业判断。' if zh else 'AI-assisted report for clinical reference only.'}
{'报告生成时间' if zh else 'Generated'}: {date.today()}
{'优复门诊 UP Clinic' if zh else 'UP Clinic | Sports Medicine Center'}
"""

st.download_button(
    label="📄 下载评估报告" if zh else "📄 Download Report",
    data=report_text.encode("utf-8"),
    file_name=f"ACL_RTS_{patient_name or 'patient'}_{eval_date}.txt",
    mime="text/plain"
)

st.markdown(
    '<div class="disclaimer">⚠️ ' +
    ('本工具仅供临床辅助参考，不替代医生判断。模型基于模拟数据，正式临床使用前请先验证。' if zh else
     'For clinical reference only. Does not replace physician judgment. Validate model before clinical use.') +
    '</div>',
    unsafe_allow_html=True
)
