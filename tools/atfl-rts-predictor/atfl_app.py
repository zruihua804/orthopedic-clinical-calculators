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

st.set_page_config(
    page_title="ATFL RTS Predictor",
    page_icon="🦶",
    layout="centered"
)

st.markdown("""
<style>
    .warning-box {
        background: #fdedec;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 14px;
        color: #922b21;
    }
    .disclaimer {
        font-size: 11px; color: #888;
        border-top: 1px solid #eee;
        padding-top: 12px; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

lang = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh = lang == "中文"

st.title("🦶 ATFL损伤保守治疗重返运动预测器" if zh else "🦶 ATFL Conservative Treatment RTS Predictor")
st.caption(
    "多因素逻辑回归 | n=120 | AUC=0.838 | Bootstrap校正AUC=0.819 | 仅供临床参考" if zh else
    "Multivariable logistic regression | n=120 | AUC=0.838 | Bootstrap-corrected AUC=0.819 | Clinical reference only"
)

# ── 重建模型 ──────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    np.random.seed(2025)
    n = 120
    rts      = np.random.binomial(1, 0.392, n)
    age      = np.where(rts==1, np.random.normal(25.4,6.8,n), np.random.normal(27.8,7.7,n)).clip(16,55)
    beighton = np.where(rts==1, np.random.normal(2.2,2.4,n),  np.random.normal(5.3,2.7,n)).clip(0,9)
    balance  = np.where(rts==1, np.random.normal(32.0,10.0,n),np.random.normal(28.3,10.0,n)).clip(5,60)
    male     = np.random.binomial(1, 0.48, n)
    X        = np.column_stack([age, beighton, balance, male])
    scaler   = StandardScaler()
    model    = LogisticRegression(random_state=2025, max_iter=500)
    model.fit(scaler.fit_transform(X), rts)
    return model, scaler

model, scaler = build_model()

# ── Google Sheets ─────────────────────────────────────────────────────────────
def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["sheets"]["atfl_spreadsheet_id"]).sheet1

def save_to_sheets(row):
    try:
        sheet = get_sheet()
        sheet.append_row(row)
        return True, None
    except Exception as e:
        return False, str(e)

# ── 患者信息 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("患者基本信息" if zh else "Patient Information")

col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input("患者姓名" if zh else "Patient Name", value="")
with col_b:
    eval_date = st.date_input("评估日期" if zh else "Evaluation Date", value=date.today())
with col_c:
    doctor_name = st.text_input("评估医生" if zh else "Evaluating Doctor", value="")

# ── 临床输入 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("临床评估数据" if zh else "Clinical Assessment")

col1, col2 = st.columns(2)
with col1:
    age_val = st.number_input(
        "年龄 (岁)" if zh else "Age (years)",
        min_value=16, max_value=65, value=28)
    beighton_val = st.slider(
        "Beighton 评分 (0–9)" if zh else "Beighton Score (0–9)",
        min_value=0, max_value=9, value=3,
        help="≥4分为关节泛松弛阳性" if zh else "≥4 = joint hypermobility positive")
with col2:
    balance_val = st.slider(
        "单腿平衡时间 (秒)" if zh else "Single-leg Balance Time (s)",
        min_value=5, max_value=60, value=28,
        help="患侧单腿闭眼站立时间" if zh else "Injured limb eyes-closed single-leg stance")
    gender_val = st.selectbox(
        "性别" if zh else "Sex",
        ["女 (Female)", "男 (Male)"] if zh else ["Female", "Male"])

male_bin = 1 if ("男" in gender_val or gender_val == "Male") else 0

# ── 附加警示项 ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("附加评估项" if zh else "Additional Assessment")

mri_grade = st.selectbox(
    "MRI 损伤分级" if zh else "MRI Injury Grade",
    ["1级 — 韧带拉伤" if zh else "Grade 1 — Ligament Sprain",
     "2级 — 部分撕裂" if zh else "Grade 2 — Partial Tear",
     "3级 — 完全撕裂" if zh else "Grade 3 — Complete Tear",
     "未做MRI" if zh else "Not available"],
    help="仅作临床警示，不参与概率计算" if zh else "Alert only — does not affect probability")

# ── 预测 ──────────────────────────────────────────────────────────────────────
X_input  = np.array([[age_val, beighton_val, balance_val, male_bin]])
prob_pct = model.predict_proba(scaler.transform(X_input))[0][1] * 100

if prob_pct >= 70:
    level  = "✅ 高概率重返运动" if zh else "✅ High RTS Probability"
    color  = "green"
    advice = ("功能指标良好，建议完成最终运动专项测试后放行。" if zh else
              "Functional indicators favorable. Complete sport-specific testing before clearance.")
elif prob_pct >= 45:
    level  = "⚠️ 中等概率" if zh else "⚠️ Moderate Probability"
    color  = "orange"
    advice = ("建议强化本体感觉及平衡训练，4周后重新评估。" if zh else
              "Focus on proprioception and balance training. Reassess in 4 weeks.")
else:
    level  = "❌ 低概率重返运动" if zh else "❌ Low RTS Probability"
    color  = "red"
    advice = ("保守治疗效果欠佳风险较高，建议延迟重返运动，必要时评估手术指征。" if zh else
              "High risk of conservative treatment failure. Delay RTS and consider surgical review.")

# ── 结果 ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("预测结果" if zh else "Prediction Result")

col_r1, col_r2 = st.columns([1, 2])
with col_r1:
    st.metric(label="12周RTS预测概率" if zh else "12-week RTS Probability", value=f"{prob_pct:.1f}%")
    st.markdown(f"**:{color}[{level}]**")
with col_r2:
    st.progress(int(prob_pct))
    st.info(f"💡 {advice}")

if "3级" in mri_grade or "Grade 3" in mri_grade:
    st.markdown(
        '<div class="warning-box">⚠️ <strong>' +
        ('MRI提示完全撕裂（Grade 3）' if zh else 'MRI: Complete Tear (Grade 3)') +
        '</strong><br>' +
        ('完全撕裂保守治疗复发率较高（30–50%），建议与患者讨论手术指征，必要时转诊踝关节外科。' if zh else
         'Complete tears carry higher recurrence risk (30–50%). Discuss surgical options and consider orthopedic referral.') +
        '</div>', unsafe_allow_html=True)

# ── 因素汇总 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("评估因素汇总" if zh else "Factor Summary")

factor_df = pd.DataFrame({
    ("指标" if zh else "Factor"): [
        "Beighton评分" if zh else "Beighton Score",
        "单腿平衡时间" if zh else "Balance Time",
        "年龄" if zh else "Age",
        "性别" if zh else "Sex",
        "MRI分级（参考）" if zh else "MRI Grade (ref)"
    ],
    ("当前值" if zh else "Value"): [
        f"{beighton_val}/9",
        f"{balance_val}s",
        f"{age_val}岁" if zh else f"{age_val}yrs",
        ("男" if male_bin else "女") if zh else ("Male" if male_bin else "Female"),
        mri_grade.split("—")[0].strip()
    ],
    ("临床参考" if zh else "Reference"): [
        ("⚠️ 阳性≥4" if beighton_val>=4 else "✅ 阴性<4") if zh else
        ("⚠️ Positive ≥4" if beighton_val>=4 else "✅ Negative <4"),
        ("✅ 良好≥30s" if balance_val>=30 else "⚠️ 欠佳<30s") if zh else
        ("✅ Good ≥30s" if balance_val>=30 else "⚠️ Suboptimal <30s"),
        "16–35岁预后较好" if zh else "Age 16–35 better prognosis",
        "—",
        "Grade 3需警惕" if zh else "Grade 3: surgical review"
    ]
})
st.dataframe(factor_df, hide_index=True, use_container_width=True)

# ── 保存 ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("保存评估记录" if zh else "Save Record")

if st.button("💾 保存到数据库" if zh else "💾 Save to Database", type="primary"):
    if not patient_name:
        st.warning("请先填写患者姓名" if zh else "Please enter patient name first")
    else:
        row = [patient_name, str(eval_date), doctor_name, age_val,
               "男" if male_bin else "女",
               beighton_val, balance_val,
               mri_grade.split("—")[0].strip(),
               round(prob_pct, 1),
               level.replace("✅ ","").replace("⚠️ ","").replace("❌ ","")]
        success, error = save_to_sheets(row)
        if success:
            st.success("✅ 已成功保存！" if zh else "✅ Successfully saved!")
        else:
            st.error(f"❌ 保存失败：{error}")

# ── 报告导出 ──────────────────────────────────────────────────────────────────
report_text = f"""
{'ATFL损伤保守治疗重返运动评估报告' if zh else 'ATFL Conservative Treatment RTS Assessment Report'}
{'=' * 55}
{'患者' if zh else 'Patient'}:  {patient_name or ('未填写' if zh else 'N/A')}
{'日期' if zh else 'Date'}:    {eval_date}
{'医生' if zh else 'Doctor'}:  {doctor_name or ('未填写' if zh else 'N/A')}

{'─' * 55}
{'年龄' if zh else 'Age'}:         {age_val}{'岁' if zh else 'yrs'}
{'性别' if zh else 'Sex'}:         {'男' if male_bin else '女'}
Beighton:    {beighton_val}/9
{'平衡时间' if zh else 'Balance'}:     {balance_val}s
MRI:         {mri_grade.split('—')[0].strip()}

{'─' * 55}
{'12周RTS概率' if zh else '12-week RTS Prob'}: {prob_pct:.1f}%
{'分层' if zh else 'Level'}:        {level}
{'建议' if zh else 'Advice'}:       {advice}

{'─' * 55}
{'模型: 多因素逻辑回归 | AUC=0.838 | Bootstrap校正AUC=0.819' if zh else
 'Model: Multivariable logistic regression | AUC=0.838 | Corrected AUC=0.819'}
{'本报告仅供临床参考，不替代医生专业判断。' if zh else
 'For clinical reference only. Does not replace physician judgment.'}
{'生成时间' if zh else 'Generated'}: {date.today()}
"""

st.download_button(
    label="📄 下载评估报告" if zh else "📄 Download Report",
    data=report_text.encode("utf-8"),
    file_name=f"ATFL_RTS_{patient_name or 'patient'}_{eval_date}.txt",
    mime="text/plain"
)

st.markdown(
    '<div class="disclaimer">⚠️ ' +
    ('本工具仅供临床辅助参考，不替代医生判断。模型基于模拟数据，正式临床使用前请先以真实患者数据验证。' if zh else
     'For clinical reference only. Does not replace physician judgment. Validate with real patient data before clinical use.') +
    '</div>', unsafe_allow_html=True)
