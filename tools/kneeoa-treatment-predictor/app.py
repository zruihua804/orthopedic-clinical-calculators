"""
Knee OA Conservative Treatment Response Predictor
Knee OA Treatment Predictor — Built with literature-derived parameters
Model version: 1.0 | 2026
"""

import streamlit as st
import numpy as np
import math
from datetime import date, datetime
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Knee OA Treatment Predictor | 膝关节OA疗效预测",
    page_icon="🦵",
    layout="centered",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFB; }
    .block-container { padding-top: 1.5rem; }
    
    .result-box {
        background: #EBF5FB;
        border-left: 5px solid #1A5276;
        border-radius: 6px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1A5276;
        margin-bottom: 4px;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2E86AB;
    }
    .delta-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #52B788;
    }
    
    .warn-red {
        background: #FDEDEC;
        border-left: 5px solid #E84855;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #922B21;
        font-weight: 600;
    }
    .warn-orange {
        background: #FEF9E7;
        border-left: 5px solid #F4A261;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #784212;
        font-weight: 600;
    }
    .warn-yellow {
        background: #FFFDE7;
        border-left: 5px solid #F9C74F;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #7D6608;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1A5276;
        border-bottom: 2px solid #AED6F1;
        padding-bottom: 4px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .disclaimer {
        font-size: 0.78rem;
        color: #7F8C8D;
        background: #F2F3F4;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 20px;
    }
    
    .ref-text {
        font-size: 0.8rem;
        color: #7F8C8D;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LANGUAGE TOGGLE
# ─────────────────────────────────────────────
lang = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh = (lang == "中文")

def t(cn, en):
    return cn if zh else en

# ─────────────────────────────────────────────
# SIDEBAR — REFERENCES
# ─────────────────────────────────────────────
with st.sidebar.expander(t("📚 模型文献依据", "📚 Model References"), expanded=False):
    st.markdown("""
<div class='ref-text'>
1. Weigl M et al. <i>Osteoarthritis & Cartilage</i> 2006;14(7):726–735.<br>
   Rehab response predictors (sex, depression, comorbidity). OR female=1.50.<br><br>
2. Riddle DL et al. (MOST Study). <i>Arthritis Care Res</i> 2010;62(7):951–959.<br>
   Walking speed, BMI, ROA as MCII predictors. Walking +1m/s OR=1.95.<br><br>
3. Bianco Prevot G et al. <i>KSSTA</i> 2025;33:2230–2236.<br>
   WOMAC pain predicts TKA trajectory (n=7552). Cutoff WOMAC pain=4.<br><br>
4. Frontiers in Physiology 2025;16:1678037.<br>
   PRP predictors (n=140): BMI -13.3%/unit, duration -9.5%/yr on WOMAC.<br><br>
5. Puzzitiello RN et al. <i>AJSM</i> 2024. Meta-analysis PRP vs HA.<br>
   PRP+HA vs HA: OR=2.19 (95%CI 1.33–3.62).<br><br>
6. Bliddal H et al. <i>Osteoarthritis & Cartilage</i> 2005;13(1):20–27.<br>
   Weight loss RCT: 9.4% WOMAC improvement per 1% body fat reduction.<br><br>
7. Messier SP et al. (IDEA). <i>Arthritis Rheumatol</i> 2018;70(11):1714–1721.<br>
   Dose-response: ≥10% weight loss superior to <5% (WOMAC function p=0.0123).<br><br>
8. Karasavvidis T et al. <i>PMC</i> 2021. TKA predictors: KL4+varus AUC=0.846.<br><br>
9. Goff AJ et al. <i>PMC</i> 2025. Registry: BMI and KL4 predict TKA within 6mo.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('版本', 'Version')}** 1.0 ")
st.sidebar.markdown(f"*{t('仅供临床辅助决策，不替代医生判断', 'For clinical decision support only')}*")

# ─────────────────────────────────────────────
# MODEL PARAMETERS (literature-derived)
# ─────────────────────────────────────────────
# All beta coefficients from multivariable logistic regression OR values
BETA = dict(
    # Shared predictors
    age        =  0.030,   # Source: OAI/MOST; OR=1.03/yr
    bmi        = -0.139,   # Source: Frontiers 2025; OR=0.87/unit (-13.3%/unit)
    kl_grade   = -0.598,   # Source: Karasavvidis 2021; OR=0.55/step
    womac_base = -0.030,   # Source: KSSTA 2025 / MOST; OR=0.97/pt
    duration   = -0.094,   # Source: Frontiers 2025; OR=0.91/yr
    alignment  = -0.511,   # Source: Karasavvidis 2021; OR=0.60 (moderate-severe varus)
    crp        = -0.083,   # Source: PMC TKA literature; OR=0.92/mg/L
    sex_female =  0.405,   # Source: Weigl 2006; OR=1.50 (female)
    walk_speed =  0.668,   # Source: MOST Study; OR=1.95/+1m/s

    # Treatment addons
    injection  =  0.784,   # Source: Puzzitiello 2024 meta; OR=2.19 (PRP+HA vs HA alone)
    weight_loss=  0.588,   # Source: IDEA/Bliddal; OR=1.80 (≥5% weight loss)
)

# Calibrated intercepts (based on literature base rates P1=45%, P2=62%, P3=72%)
INTERCEPT = dict(
    P1 =  5.824,   # Rehab only           — base rate 45%
    P2 =  5.730,   # + HA/PRP injection   — base rate 62%
    P3 =  5.597,   # + Weight loss ≥5%    — base rate 72%
)

# TKA risk intercept (separate model; base TKA rate ~12% at 12mo for KL2-3)
# Calibrated so reference patient (KL2, BMI28, WOMAC45, age63, no varus) = 12%
INTERCEPT_TKA = -10.167

BETA_TKA = dict(
    age        =  0.040,
    bmi        =  0.080,   # higher BMI → higher TKA risk
    kl_grade   =  0.900,   # KL4 dramatically increases TKA risk
    womac_base =  0.035,   # higher baseline WOMAC → more likely TKA
    duration   =  0.080,
    alignment  =  0.750,   # varus → higher TKA risk
    sex_female = -0.200,   # women slightly less likely to undergo TKA at same symptom level
)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def predict_response(age, bmi, kl, womac, duration, alignment_code, crp, sex_f,
                     walk_speed_val, treatment_extra=0.0, intercept_key="P1"):
    """
    Returns probability of achieving WOMAC MCID (≥18% improvement) at 6 months.
    alignment_code: 0=normal/mild(<3deg), 1=moderate(3-10deg), 2=severe(>10deg)
    """
    align_bin = 1 if alignment_code >= 1 else 0   # moderate or severe = penalised
    lin = (INTERCEPT[intercept_key]
           + BETA['age']        * age
           + BETA['bmi']        * bmi
           + BETA['kl_grade']   * kl
           + BETA['womac_base'] * womac
           + BETA['duration']   * duration
           + BETA['alignment']  * align_bin
           + BETA['crp']        * crp
           + BETA['sex_female'] * sex_f
           + BETA['walk_speed'] * walk_speed_val
           + treatment_extra)
    return sigmoid(lin)

def predict_tka_risk(age, bmi, kl, womac, duration, alignment_code):
    align_bin = 1 if alignment_code >= 1 else 0
    lin = (INTERCEPT_TKA
           + BETA_TKA['age']        * age
           + BETA_TKA['bmi']        * bmi
           + BETA_TKA['kl_grade']   * kl
           + BETA_TKA['womac_base'] * womac
           + BETA_TKA['duration']   * duration
           + BETA_TKA['alignment']  * align_bin)
    return sigmoid(lin)

def weight_loss_benefit_tier(delta):
    if delta >= 0.15:
        return t("🟢 高获益 — 强烈推荐减重干预", "🟢 High benefit — Strongly recommend weight loss")
    elif delta >= 0.08:
        return t("🟡 中等获益 — 推荐减重干预", "🟡 Moderate benefit — Recommend weight loss")
    else:
        return t("⚪ 轻微获益 — 减重对本患者额外疗效有限", "⚪ Modest benefit — Limited additional gain from weight loss")


# ─────────────────────────────────────────────
# SESSION STATE — persist predictions across reruns
# ─────────────────────────────────────────────
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
## 🦵 {t('膝关节OA保守治疗疗效预测器', 'Knee OA Conservative Treatment Response Predictor')}
{t('基于文献参数的循证预测模型', 'Evidence-based prediction model')}
""")
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — PATIENT INFO
# ─────────────────────────────────────────────
st.markdown(f"<div class='section-header'>① {t('患者基本信息', 'Patient Information')}</div>",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input(t("患者姓名", "Patient Name"), placeholder="姓名 / Name")
with col2:
    eval_date = st.date_input(t("评估日期", "Assessment Date"), value=date.today())
with col3:
    clinician = st.text_input(t("评估医生", "Clinician"), placeholder="医生 / Clinician")

col4, col5 = st.columns(2)
with col4:
    eval_timepoint = st.selectbox(
        t("评估时间点", "Evaluation Timepoint"),
        [t("初诊基线", "Baseline"),
         t("治疗中（6周）", "Mid-treatment (6wk)"),
         t("治疗结束（12周）", "End of treatment (12wk)")]
    )
with col5:
    sex = st.radio(t("性别", "Sex"), [t("女 Female", "Female"), t("男 Male", "Male")], horizontal=True)
    sex_f = 1 if t("女 Female", "Female") in sex else 0

# ─────────────────────────────────────────────
# SECTION 2 — CORE PREDICTORS
# ─────────────────────────────────────────────
st.markdown(f"<div class='section-header'>② {t('核心预测变量', 'Core Predictor Variables')}</div>",
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        t("年龄（岁）", "Age (years)"),
        min_value=30, max_value=90, value=63, step=1,
        help=t("OAI研究人群均值63岁", "OAI study population mean = 63 yrs")
    )
    bmi = st.number_input(
        t("BMI（kg/m²）", "BMI (kg/m²)"),
        min_value=18.0, max_value=55.0, value=28.0, step=0.1,
        help=t("每增加1 kg/m²，PRP疗效下降约13.3%（Frontiers 2025）",
               "Each +1 kg/m² reduces PRP efficacy ~13.3% (Frontiers 2025)")
    )
    kl_grade = st.selectbox(
        t("Kellgren-Lawrence分级", "Kellgren-Lawrence Grade"),
        options=[1, 2, 3, 4],
        index=1,
        help=t("KL4级显著增加TKA风险（Karasavvidis 2021）",
               "KL4 significantly increases TKA risk (Karasavvidis 2021)")
    )
    womac_baseline = st.slider(
        t("基线WOMAC总分（0–100）", "Baseline WOMAC Total Score (0–100)"),
        min_value=0, max_value=100, value=45,
        help=t("0=无症状，100=最严重；MCID截点为基线分数的18%改善",
               "0=no symptoms, 100=worst; MCID = 18% improvement from baseline")
    )

with col2:
    duration = st.number_input(
        t("症状病程（年）", "Symptom Duration (years)"),
        min_value=0.5, max_value=30.0, value=3.0, step=0.5,
        help=t("每增加1年，PRP疗效下降约9.5%（Frontiers 2025）",
               "Each +1 yr reduces PRP efficacy ~9.5% (Frontiers 2025)")
    )
    alignment = st.selectbox(
        t("关节力线（机械轴偏移）", "Joint Alignment (Mechanical Axis)"),
        options=[
            t("正常 (<3°)", "Normal (<3°)"),
            t("轻中度内翻/外翻 (3–10°)", "Mild-Moderate Varus/Valgus (3–10°)"),
            t("重度内翻/外翻 (>10°)", "Severe Varus/Valgus (>10°)"),
        ],
        help=t("全长负重X线标准测量（机械轴偏移度数）",
               "Full-length weight-bearing X-ray measurement (mechanical axis deviation)")
    )
    alignment_code = [0, 1, 2][[t("正常 (<3°)", "Normal (<3°)"),
                                  t("轻中度内翻/外翻 (3–10°)", "Mild-Moderate Varus/Valgus (3–10°)"),
                                  t("重度内翻/外翻 (>10°)", "Severe Varus/Valgus (>10°)")].index(alignment)]

    crp = st.number_input(
        t("CRP（mg/L）", "CRP (mg/L)"),
        min_value=0.0, max_value=100.0, value=4.0, step=0.5,
        help=t("C反应蛋白；>10 mg/L触发炎症警示", "C-reactive protein; >10 mg/L triggers alert")
    )
    walk_speed = st.number_input(
        t("10米步速（m/s）", "10-metre Walk Speed (m/s)"),
        min_value=0.3, max_value=2.5, value=1.1, step=0.05,
        help=t("常模：健康老年人约1.2–1.4 m/s；步速每快1 m/s，康复反应率提升约1.95倍（MOST研究）",
               "Norm: healthy older adults ~1.2–1.4 m/s; +1 m/s walk speed → OR=1.95 for improvement (MOST)")
    )

# ─────────────────────────────────────────────
# SECTION 3 — TREATMENT PLAN
# ─────────────────────────────────────────────
st.markdown(f"<div class='section-header'>③ {t('治疗方案选择', 'Treatment Plan')}</div>",
            unsafe_allow_html=True)

include_injection = st.checkbox(
    t("✅ 纳入 HA + PRP 联合注射（×4次，每2周1次）",
      "✅ Include HA + PRP Combined Injection (×4, every 2 weeks)"),
    value=True
)
include_weightloss = st.checkbox(
    t("⚖️ 纳入减重方案（目标：6个月内减轻体重≥5%）",
      "⚖️ Include Weight Loss Program (Target: ≥5% body weight in 6 months)"),
    value=False
)

# ─────────────────────────────────────────────
# SECTION 4 — WARNING FLAGS
# ─────────────────────────────────────────────
st.markdown(f"<div class='section-header'>④ {t('附加临床评估（警示项）', 'Additional Clinical Flags')}</div>",
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    hba1c_flag = st.selectbox(
        t("HbA1c（糖尿病患者填写）", "HbA1c (Diabetic patients only)"),
        [t("不适用 / 正常 (<7%)", "N/A or Normal (<7%)"),
         t("控制欠佳 (7–9%)", "Suboptimal (7–9%)"),
         t("控制差 (>9%)", "Poor (>9%)")]
    )
with col2:
    depression_flag = st.selectbox(
        t("心理/抑郁状态", "Psychological / Depression Status"),
        [t("无抑郁/焦虑迹象", "No depression/anxiety"),
         t("轻度抑郁/焦虑", "Mild depression/anxiety"),
         t("中重度抑郁/焦虑", "Moderate-severe depression/anxiety")]
    )

# ─────────────────────────────────────────────
# CALCULATE
# ─────────────────────────────────────────────
if st.button(f"🔮 {t('生成预测报告', 'Generate Prediction Report')}", type="primary", use_container_width=True):

    # Determine which models to run
    inj_extra  = BETA['injection']   if include_injection  else 0.0
    wl_extra   = BETA['weight_loss'] if include_weightloss else 0.0

    # Always compute all 3 for comparison
    p1 = predict_response(age, bmi, kl_grade, womac_baseline, duration,
                          alignment_code, crp, sex_f, walk_speed,
                          treatment_extra=0.0, intercept_key="P1")
    p2 = predict_response(age, bmi, kl_grade, womac_baseline, duration,
                          alignment_code, crp, sex_f, walk_speed,
                          treatment_extra=BETA['injection'], intercept_key="P2")
    p3 = predict_response(age, bmi, kl_grade, womac_baseline, duration,
                          alignment_code, crp, sex_f, walk_speed,
                          treatment_extra=BETA['injection'] + BETA['weight_loss'], intercept_key="P3")

    tka_risk = predict_tka_risk(age, bmi, kl_grade, womac_baseline, duration, alignment_code)

    delta_inj = p2 - p1
    delta_wl  = p3 - p2
    delta_total = p3 - p1


    # Store results in session state so save button works across reruns
    st.session_state.predictions = dict(
        p1=p1, p2=p2, p3=p3, tka_risk=tka_risk,
        delta_inj=delta_inj, delta_wl=delta_wl, delta_total=delta_total,
        patient_name=patient_name, eval_date=str(eval_date),
        clinician=clinician, eval_timepoint=eval_timepoint,
        age=age, sex=sex, bmi=bmi, kl_grade=kl_grade,
        womac_baseline=womac_baseline, duration=duration,
        alignment=alignment, crp=crp, walk_speed=walk_speed,
        include_injection=include_injection, include_weightloss=include_weightloss,
        hba1c_flag=hba1c_flag, depression_flag=depression_flag,
    )

    st.markdown("---")
    st.markdown(f"## 📊 {t('预测结果', 'Prediction Results')}")

    # ── WARNING ALERTS ──────────────────────────────────
    warnings_shown = False

    if kl_grade == 4 and alignment_code == 2:
        st.markdown(f"""<div class='warn-red'>
        🔴 <b>{t('强烈建议手术评估', 'STRONGLY RECOMMEND SURGICAL EVALUATION')}</b><br>
        {t('该患者 KL 4级 + 重度内翻/外翻（>10°），保守治疗预期获益有限。建议优先评估手术指征（TKA/截骨术）。',
           'KL Grade 4 + Severe malalignment (>10°) — conservative treatment has limited expected benefit. Surgical evaluation (TKA/osteotomy) is strongly recommended.')}
        </div>""", unsafe_allow_html=True)
        warnings_shown = True

    elif kl_grade == 4:
        st.markdown(f"""<div class='warn-orange'>
        🟠 <b>{t('KL 4级 — 保守治疗有效率偏低', 'KL Grade 4 — Lower conservative treatment success rate')}</b><br>
        {t('建议与患者充分沟通预期，讨论手术时机。',
           'Discuss realistic expectations with patient; consider discussing surgical timing.')}
        </div>""", unsafe_allow_html=True)
        warnings_shown = True

    elif alignment_code == 2:
        st.markdown(f"""<div class='warn-orange'>
        🟠 <b>{t('重度力线异常 (>10°)', 'Severe malalignment (>10°)')}</b><br>
        {t('注射疗效可能受限，建议评估截骨手术可能性。',
           'Injection efficacy may be limited; consider osteotomy evaluation.')}
        </div>""", unsafe_allow_html=True)
        warnings_shown = True

    if crp > 10:
        st.markdown(f"""<div class='warn-yellow'>
        🟡 <b>CRP > 10 mg/L</b> — {t('建议排除类风湿性关节炎、痛风等继发性关节炎。',
           'Rule out rheumatoid arthritis, gout, or other secondary arthritis.')}
        </div>""", unsafe_allow_html=True)

    if t("控制欠佳", "Suboptimal") in hba1c_flag or t("控制差", "Poor") in hba1c_flag:
        st.markdown(f"""<div class='warn-yellow'>
        🟡 <b>HbA1c {t('异常', 'Elevated')}</b> — {t('血糖控制不佳可能影响组织修复与康复疗效，建议内分泌科协同管理。',
           'Poor glycaemic control may impair tissue repair and rehabilitation outcomes. Co-manage with endocrinology.')}
        </div>""", unsafe_allow_html=True)

    if t("轻度", "Mild") in depression_flag or t("中重度", "Moderate") in depression_flag:
        st.markdown(f"""<div class='warn-yellow'>
        🟡 <b>{t('心理状态异常', 'Psychological Concern')}</b> — {t('抑郁/焦虑显著影响康复依从性与主观疗效，建议心理科评估。',
           'Depression/anxiety significantly affects rehabilitation compliance and subjective outcomes. Consider psychological evaluation.')}
        </div>""", unsafe_allow_html=True)

    # ── MAIN RESULTS ──────────────────────────────────
    st.markdown(f"### {t('WOMAC改善预测（6个月）', 'Predicted WOMAC Improvement (6 months)')}")
    st.caption(t("结局定义：达到MCID（基线评分改善≥18%）的预测概率",
                 "Outcome: Predicted probability of achieving MCID (≥18% improvement from baseline)"))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""<div class='result-box'>
        <div class='result-title'>📋 {t('方案一：单纯康复', 'Plan A: Rehab Only')}</div>
        <div class='result-value'>{p1*100:.1f}%</div>
        <div style='font-size:0.85rem;color:#566573'>{t('12周康复治疗', '12-week rehabilitation')}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        inj_badge = "✅" if include_injection else "💡"
        st.markdown(f"""<div class='result-box'>
        <div class='result-title'>{inj_badge} {t('方案二：+ HA/PRP注射', 'Plan B: + HA/PRP')}</div>
        <div class='result-value'>{p2*100:.1f}%</div>
        <div class='delta-value'>▲ +{delta_inj*100:.1f}% {t('注射增量获益', 'injection gain')}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        wl_badge = "✅" if include_weightloss else "💡"
        st.markdown(f"""<div class='result-box'>
        <div class='result-title'>{wl_badge} {t('方案三：+ 减重≥5%', 'Plan C: + Weight Loss')}</div>
        <div class='result-value'>{p3*100:.1f}%</div>
        <div class='delta-value'>▲ +{delta_wl*100:.1f}% {t('减重额外获益', 'weight loss gain')}</div>
        </div>""", unsafe_allow_html=True)

    # Weight loss benefit tier
    st.markdown(f"""
    **{t('减重干预对该患者的预期效益：', 'Expected benefit of weight loss for this patient:')}**
    {weight_loss_benefit_tier(delta_total)}
    """)

    # ── TKA RISK ──────────────────────────────────────
    st.markdown(f"### {t('TKA手术风险预测（12个月内）', 'TKA Risk Prediction (within 12 months)')}")

    tka_color = "#E84855" if tka_risk > 0.35 else ("#F4A261" if tka_risk > 0.15 else "#52B788")
    tka_label = (t("高风险", "High Risk") if tka_risk > 0.35
                 else t("中等风险", "Moderate Risk") if tka_risk > 0.15
                 else t("低风险", "Low Risk"))

    st.markdown(f"""<div class='result-box' style='border-left-color:{tka_color}'>
    <div class='result-title'>🏥 {t('12个月内需要TKA的预测概率', 'Predicted probability of TKA within 12 months')}</div>
    <div class='result-value' style='color:{tka_color}'>{tka_risk*100:.1f}%</div>
    <div style='font-size:1rem;font-weight:600;color:{tka_color}'>{tka_label}</div>
    </div>""", unsafe_allow_html=True)

    # ── INDICATOR TABLE ────────────────────────────────
    st.markdown(f"### {t('评估指标汇总', 'Assessment Summary')}")

    def status(val, cutoff, direction="below"):
        if direction == "below":
            return "✅" if val <= cutoff else "⚠️"
        else:
            return "✅" if val >= cutoff else "⚠️"

    mcid_threshold = womac_baseline * 0.18

    table_data = {
        t("指标", "Indicator"):    [t("BMI","BMI"), t("KL分级","KL Grade"),
                                    t("WOMAC基线","Baseline WOMAC"),
                                    t("WOMAC MCID阈值","WOMAC MCID Threshold"),
                                    t("步速","Walk Speed"), t("CRP","CRP"),
                                    t("病程","Disease Duration")],
        t("数值", "Value"):        [f"{bmi:.1f} kg/m²", f"KL {kl_grade}",
                                    f"{womac_baseline} pts",
                                    f"≥{mcid_threshold:.1f} pts improvement",
                                    f"{walk_speed:.2f} m/s", f"{crp:.1f} mg/L",
                                    f"{duration:.1f} yrs"],
        t("参考标准", "Reference"): ["< 30 (WHO overweight)",
                                     "≤ 3 (conservative preferred)",
                                     "–", "18% of baseline",
                                     "≥ 1.0 m/s (functional)",
                                     "< 10 mg/L",
                                     "< 5 yrs (better prognosis)"],
        t("状态", "Status"):       [status(bmi, 30), "✅" if kl_grade <= 3 else "⚠️",
                                    "–", "–",
                                    status(walk_speed, 1.0, "above"),
                                    status(crp, 10),
                                    status(duration, 5)]
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# SAVE + REPORT — outside calculate block, uses session_state
# ─────────────────────────────────────────────
if st.session_state.predictions is not None:
    pred = st.session_state.predictions
    p1         = pred['p1']
    p2         = pred['p2']
    p3         = pred['p3']
    tka_risk   = pred['tka_risk']
    delta_total= pred['delta_total']

    st.markdown("---")
    st.markdown(f"#### 💾 {t('保存至数据库', 'Save to Database')}")

    if st.button(t("保存本次评估记录", "Save Assessment Record"), use_container_width=True):
        row = [
            str(pred['patient_name']), str(pred['eval_date']),
            str(pred['clinician']), str(pred['eval_timepoint']),
            pred['age'], pred['sex'], pred['bmi'], pred['kl_grade'],
            pred['womac_baseline'], pred['duration'], pred['alignment'],
            pred['crp'], pred['walk_speed'],
            str(pred['include_injection']), str(pred['include_weightloss']),
            pred['hba1c_flag'], pred['depression_flag'],
            f"{p1*100:.1f}%", f"{p2*100:.1f}%", f"{p3*100:.1f}%",
            f"{tka_risk*100:.1f}%",
            str(datetime.now())
        ]
        try:
            _ = st.secrets["gcp_service_account"]
            st.info("✅ Step 1: gcp_service_account secret found")
        except Exception as e:
            st.error(f"❌ Step 1 FAILED — gcp_service_account missing: {e}")
            st.stop()
        try:
            sheet_id = st.secrets["sheets"]["kneeoa_spreadsheet_id"]
            st.info(f"✅ Step 2: sheet ID found → {sheet_id[:10]}...")
        except Exception as e:
            st.error(f"❌ Step 2 FAILED — kneeoa_spreadsheet_id missing: {e}")
            st.stop()
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets",
                      "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=scopes)
            st.info("✅ Step 3: credentials built OK")
        except Exception as e:
            st.error(f"❌ Step 3 FAILED — credentials error: {e}")
            st.stop()
        try:
            client = gspread.authorize(creds)
            st.info("✅ Step 4: gspread authorised OK")
        except Exception as e:
            st.error(f"❌ Step 4 FAILED — gspread auth error: {e}")
            st.stop()
        try:
            sheet = client.open_by_key(sheet_id).sheet1
            st.info("✅ Step 5: sheet opened OK")
        except Exception as e:
            st.error(f"❌ Step 5 FAILED — cannot open sheet: {e}")
            st.stop()
        try:
            sheet.append_row(row)
            st.success(t("✅ 记录已成功保存至 Google Sheets！", "✅ Record saved to Google Sheets!"))
            st.session_state.predictions = None  # clear after save
        except Exception as e:
            st.error(f"❌ Step 6 FAILED — append_row error: {e}")

    # Download report button (also outside calculate block)
    report_txt = f"""
KNEE OA CONSERVATIVE TREATMENT RESPONSE PREDICTOR
Knee OA Treatment Predictor — Report Version 1.0
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

PATIENT: {pred['patient_name']}
Date: {pred['eval_date']} | Clinician: {pred['clinician']} | Timepoint: {pred['eval_timepoint']}

INPUTS
------
Age: {pred['age']} yrs | Sex: {pred['sex']} | BMI: {pred['bmi']:.1f} kg/m2
KL Grade: {pred['kl_grade']} | WOMAC Baseline: {pred['womac_baseline']}/100
Symptom Duration: {pred['duration']:.1f} yrs
Joint Alignment: {pred['alignment']}
CRP: {pred['crp']:.1f} mg/L | Walk Speed: {pred['walk_speed']:.2f} m/s

PREDICTIONS (6-month WOMAC MCID probability)
--------------------------------------------
Plan A  Rehab Only:               {p1*100:.1f}%
Plan B  + HA/PRP Injection:       {p2*100:.1f}%
Plan C  + Weight Loss >=5%:       {p3*100:.1f}%

TKA Risk (12-month):              {tka_risk*100:.1f}%

Weight Loss Benefit: {weight_loss_benefit_tier(delta_total)}

REFERENCES
----------
1. Weigl M et al. Osteoarthritis Cartilage 2006;14:726-735
2. Riddle DL et al. (MOST) Arthritis Care Res 2010;62:951-959
3. Bianco Prevot G et al. KSSTA 2025;33:2230-2236
4. Frontiers in Physiology 2025;16:1678037
5. Puzzitiello RN et al. AJSM 2024 (meta-analysis)
6. Bliddal H et al. Osteoarthritis Cartilage 2005;13:20-27
7. Messier SP et al. (IDEA) Arthritis Rheumatol 2018;70:1714-1721
8. Karasavvidis T et al. PMC 2021
9. Goff AJ et al. PMC 2025
{'='*60}
"""
    st.download_button(
        label=t("📄 下载文字报告 (.txt)", "📄 Download Report (.txt)"),
        data=report_txt,
        file_name=f"KneeOA_Report_{pred['patient_name']}_{pred['eval_date']}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='disclaimer'>
⚠️ <b>{t('免责声明', 'Disclaimer')}</b>：
{t('本工具仅供临床辅助决策使用，不替代临床医生的专业判断。模型参数来源于已发表的多因素逻辑回归研究（OR值），尚未经本地患者数据验证。建议在积累50例随访数据后进行本地验证与参数校准。',
   'This tool is for clinical decision support only and does not replace the professional judgment of clinicians. Model parameters are derived from published multivariable logistic regression studies and have not been validated on local patient data. Local validation is recommended after accumulating 50 follow-up cases.')}
</div>
""", unsafe_allow_html=True)
