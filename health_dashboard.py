# [T-SUM_2026-1_Proj.ipynb] AI 심혈관 질환 예측 시스템 대시보드 (Streamlit)
# 2026.05.07. 최종 수정

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- 1. 페이지 기본 설정 ---
st.set_page_config(
    page_title="통합 헬스케어 AI 포털",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 다중 모델 로드 및 학습 함수 (캐싱) ---
@st.cache_resource
def load_heart_model():
    df = pd.read_csv('heart.csv')
    df = df.fillna(df.mean(numeric_only=True))
    # one-hot 인코딩 (범주형 변수)
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, X.columns, df

# 향후 멘티들이 추가할 모델을 위한 자리 (뼈대 구축)
@st.cache_resource
def load_diabetes_model():
    # TODO: 당뇨병 데이터 로드 및 학습 로직 추가
    pass

@st.cache_resource
def load_stroke_model():
    # TODO: 뇌졸중 데이터 로드 및 학습 로직 추가
    pass

# 심장병 모델 자원 로드
heart_model, heart_scaler, heart_cols, heart_df = load_heart_model()


# --- 3. 사이드바 UI ---
st.sidebar.title("🏥 종합 진단 센터")
disease_category = st.sidebar.radio("진단 과목 선택", ["🫀 심혈관 질환 예측", "🩸 당뇨병 (준비중)", "🧠 뇌졸중 (준비중)"])

st.sidebar.divider()

# --- 4. 개별 질병 로직 (심혈관 질환 선택 시) ---
if disease_category == "🫀 심혈관 질환 예측":
    st.sidebar.header("📋 환자 임상 정보 입력")
    
    # 기본 정보
    age = st.sidebar.slider("나이 (Age)", 20, 100, 50)
    sex = st.sidebar.selectbox("성별 (Sex)", ['M', 'F'])
    chest_pain = st.sidebar.selectbox(
    "가슴 통증 유형", 
    ['TA', 'ATA', 'NAP', 'ASY'],
    help="""
    - TA: 전형적인 가슴 통증
    - ATA: 비전형적 통증
    - NAP: 심장과 무관해 보이는 통증
    - ASY: 무증상 (통증은 없으나 검사상 이상 소견)
    * 잘 모르시겠다면 '정상' 범위인 TA나 NAP를 선택해 보세요.
    """
    )
    resting_bp = st.sidebar.number_input("안정시 혈압 (RestingBP)", 80, 200, 120, help="편안하게 쉬고 있을 때 측정한 수축기 혈압입니다.\n\n정상 수치는 보통 120 이하입니다.")
    cholesterol = st.sidebar.number_input("혈청 콜레스테롤 (Cholesterol)", 0, 600, 200, help="혈액 내 콜레스테롤 수치(mg/dl)입니다.\n\n200 이상일 경우 주의가 필요합니다.")
    fasting_bs = st.sidebar.selectbox("공복 혈당 > 120 mg/dl (FastingBS)", [0, 1], help="1: 공복혈당 120 초과(당뇨 의심)\n\n0: 정상")
    max_hr = st.sidebar.slider("최대 심박수 (MaxHR)", 60, 202, 150, help="운동 중 도달한 가장 높은 심박수입니다.")
    exercise_angina = st.sidebar.selectbox("운동 유발성 협심증 (ExerciseAngina)", ['Y', 'N'], help="운동을 할 때 흉통이 유발되는지 여부입니다.")

    # 정밀 검사 정보 (Expander)
    with st.sidebar.expander("🩺 정밀 심전도(ECG) 검사 기록이 있다면 펼쳐주세요"):
        st.info("검사 기록을 모르신다면, 건강한 상태(정상치)를 가정하고 진단을 진행합니다.")
        know_ecg = st.checkbox("심전도 검사 결과를 알고 있습니다.")
        
        if know_ecg:
            resting_ecg = st.selectbox(
                "안정시 심전도 (RestingECG)", 
                ['Normal', 'ST', 'LVH'],
                help="Normal: 정상\n\nST: 심전도 파형 이상(ST-T파 이상)\n\nLVH: 좌심실 비대 의심 소견입니다."
            )
            st_slope_label = st.selectbox(
                "운동 시 심장 압박감 정도 (ST Slope)",
                ["운동 시에도 가뿐함 (정상)", "운동 시 가슴에 무거운 압박감이나 뻐근함이 지속됨", "운동 시 쥐어짜는 듯한 통증이나 심한 호흡 곤란 발생"],
                help="운동 강도가 높을 때 심장 근육의 혈류 반응을 나타냅니다."
            )
            
            # 내부 매핑 로직
            slope_map = {
                "운동 시에도 가뿐함 (정상)": "Up",
                "운동 시 가슴에 무거운 압박감이나 뻐근함이 지속됨": "Flat",
                "운동 시 쥐어짜는 듯한 통증이나 심한 호흡 곤란 발생": "Down"
            }
            st_slope = slope_map[st_slope_label]
            
            oldpeak = st.slider(
                "ST 하강 수치 (Oldpeak)", 
                -2.0, 6.0, 0.0, 0.1,
                help="휴식 대비 운동 시 심전도가 얼마나 낮아졌는지 나타내는 수치입니다.\n\n수치가 높을수록 심장 부담이 큼을 의미합니다."
            )
        else:
            # 사용자가 모를 경우 의학적 정상(Baseline) 값 강제 할당
            resting_ecg = 'Normal'
            st_slope = 'Up'
            oldpeak = 0.0

    # --- 메인 화면 렌더링 ---
    st.title("🫀 다중 모달리티 바이오/헬스케어 예측 시스템")
    st.markdown("현재 **[심혈관 질환 부문]** 진단 모듈이 가동 중입니다. 좌측의 임상 데이터를 변경하여 실시간 예측을 확인하세요.")
    with st.expander("🚨 이용 전 필독: 의료적 면책 조항 확인", expanded=True):
        st.markdown("""
        <div style="border: 2px solid #ff4b4b; padding: 15px; border-radius: 5px; background-color: #fff1f1;">
            <h4 style="color: #ff4b4b; margin-top: 0;">⚠️ 의료적 면책 조항</h4>
            <p style="font-size: 0.9em; color: #31333f; line-height: 1.6;">
                본 시스템은 <strong>AI 기반 참고용 건강관리 도구</strong>입니다.<br>
                질병의 진단이나 처방을 목적으로 하지 않으며, <span style="color:red; font-weight:bold;">전문 의료진의 진단을 대체할 수 없습니다.</span><br><br>
                이상 증상 발생 시 <span style="color:red; font-weight:bold;">반드시 의료기관을 방문</span>하셔야 하며, 
                응급 상황 시에는 <span style="color:red; font-weight:bold;">즉시 119에 연락</span>하시기 바랍니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

    user_input_data = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
        'MaxHR': max_hr, 'Oldpeak': oldpeak, 'Sex': sex, 'ChestPainType': chest_pain.split(" ")[0], # 한글 설명 제외하고 코드(ASY 등)만 추출
        'RestingECG': resting_ecg, 'ExerciseAngina': exercise_angina, 'ST_Slope': st_slope
    }
    user_df = pd.DataFrame([user_input_data])
    user_encoded = pd.get_dummies(user_df)
    user_encoded = user_encoded.reindex(columns=heart_cols, fill_value=0)
    user_scaled = heart_scaler.transform(user_encoded)

    prediction = heart_model.predict(user_scaled)[0]
    prediction_proba = heart_model.predict_proba(user_scaled)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("진단 결과")
        if prediction == 1:
            st.error("🚨 **위험 (High Risk)**: 심혈관 질환 발병 가능성이 높습니다.")
        else:
            st.success("✅ **정상 (Low Risk)**: 심혈관 질환 발병 가능성이 낮습니다.")
        st.metric(label="AI 예측 양성 확률", value=f"{prediction_proba * 100:.1f}%")
        st.progress(prediction_proba)
        st.divider()
        st.caption("""
        :red[**⚠️ 의료적 면책 조항 (Medical Disclaimer)**]\n
        본 시스템은 입력된 건강 데이터를 바탕으로 질병 위험도를 예측하고 생활습관 개선 정보를 제공하는 :red[**AI 기반 참고용 건강관리 도구**]입니다.\n
        본 결과는 질병의 진단, 치료, 예방 또는 의학적 처방을 목적으로 하지 않으며, :red[**전문 의료진의 진단이나 상담을 대체하지 않습니다.**]\n
        예측 결과가 낮게 나타나더라도 흉통, 호흡 곤란, 어지러움, 두근거림 등 이상 증상이 지속되면 :red[**반드시 의료기관을 방문하시기 바랍니다.**]\n
        예측 결과가 높게 나타난 경우에도 :red[**실제 질병 여부는 전문의의 진료와 정밀 검사를 통해 확인되어야 합니다.**]\n
        AI 예측 및 권고안은 :red[**부정확**]하거나 :red[**불완전**]할 수 있으며, 응급상황에서는 본 시스템을 사용하지 말고 :red[**즉시 119 또는 가까운 응급의료기관에 연락하시기 바랍니다.**]
        """)

    with col2:
        st.subheader("🕵️ AI 심층 분석 리포트 (XAI)")
        
        # 그래프 해석 가이드
        with st.expander("❓ 그래프 수치와 기호가 궁금하신가요?"):
            st.markdown("""
            **변수 옆 좌측의 숫자 (예: 1.151)**
            - 인공지능이 판단을 내리기 위해 변환한 **상대적 위험 점수**입니다. 
            - 0보다 크면 평균보다 높은 상태, 작으면 낮은 상태를 의미합니다.
            
            **막대 속 숫자 및 색상**
            - **빨간색 (+)**: 심장병 위험도를 **높이는** 부정적 신호입니다.
            - **파란색 (-)**: 심장병 위험도를 **낮추는** 긍정적(건강한) 신호입니다.
            - **맨 아래 E[f(x)]**: 데이터셋 전체의 **평균 발병 확률**(출발점)입니다.
            - **맨 위 f(x)**: 모든 요인을 더하고 뺀 **최종 위험 확률**입니다.
            """)

        # 1. SHAP 설명 객체 생성
        explainer = shap.Explainer(heart_model)
        
        # 2. 환자의 입력 데이터를 DataFrame으로 묶어 변수명을 보존합니다.
        user_scaled_df = pd.DataFrame(user_scaled, columns=heart_cols)
        
        # 3. 모델의 결과(predict)가 아닌, '입력값(user_scaled_df)' 자체를 넣습니다.
        shap_values_single = explainer(user_scaled_df)

        # 4. 그래프 생성 (Matplotlib과 SHAP 결합)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 심장병 위험(Class 1)에 대한 Waterfall 플롯 생성
        shap.plots.waterfall(shap_values_single[0, :, 1], max_display=10, show=False)
        
        # 폰트 크기 및 레이아웃 자동 조정
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("▲ 오른쪽(빨간색)으로 향하는 막대는 위험도를 높이는 요인, 왼쪽(파란색)은 낮추는 요인입니다.")

        
        # --- 의학적 소견 번역 엔진 (Mapping Engine) ---
        medical_mapping = {
            "ST_Slope_Up": "운동 시 심장 박동 후 회복 반응이 매우 건강한 상태 (정상 상승)",
            "ST_Slope_Flat": "운동 시 심전도 파형이 평탄함 (심장 혈류 공급 부족 가능성)",
            "ST_Slope_Down": "운동 시 심전도 파형이 하강함 (심근 허혈 가능성 높음)",
            "ChestPainType_ASY": "통증을 느끼지 못하는 '무증상' 가슴 답답함 (가장 주의 필요)",
            "ChestPainType_ATA": "비전형적인 가슴 통증 (심장 질환 가능성 낮음)",
            "ExerciseAngina_N": "운동 중 협심증(가슴 조임) 증상이 나타나지 않음",
            "ExerciseAngina_Y": "운동 중 가슴 통증 발생 (심혈관 협착 의심)",
            "Oldpeak": "운동 시 심전도 하강 수치 (수치가 높을수록 심장 부담 큼)",
            "MaxHR": "최대 심박수 (수치가 높을수록 심장 펌핑 능력이 양호함)",
            "Age": "연령에 따른 생물학적 위험도",
            "Cholesterol": "혈중 콜레스테롤 수치",
            "RestingBP": "안정시 혈압 수치"
        }

        # --- AI 서술형 소견 생성 로직 ---
        # shap_values_single[0, :, 1] 에서 값이 가장 큰(위험한) 항목 찾기
        vals = shap_values_single[0, :, 1].values
        feature_names = shap_values_single.feature_names

        # 위험 요인(양수)과 건강 요인(음수) 정렬
        top_risk_idx = np.argsort(vals)[-1] # 가장 위험한 요인
        top_healthy_idx = np.argsort(vals)[0] # 가장 건강한 요인

        # 인덱스를 이용해 '컬럼명'을 가져온 뒤, 사전에서 '번역문'을 찾음
        risk_feature_name = feature_names[top_risk_idx]
        healthy_feature_name = feature_names[top_healthy_idx]
        
        # 번역된 메시지 가져오기 (사전에 없으면 원래 이름 사용)
        risk_msg = medical_mapping.get(risk_feature_name, risk_feature_name)
        healthy_msg = medical_mapping.get(healthy_feature_name, healthy_feature_name)

        st.divider()
        st.info(f"""
        **📢 AI 종합 진단 소견**
        * **주의가 필요한 요인**: 현재 입력하신 정보 중 **'{risk_feature_name} : {risk_msg}'** 항목이 당신의 심장 건강 위험을 높이는 가장 큰 원인입니다.
        * **긍정적인 건강 요인**: 반면 **'{healthy_feature_name} : {healthy_msg}'** 항목은 현재 당신의 심장 건강을 유지하는 데 가장 긍정적인 기여를 하고 있습니다.
        """)

        # 리포트 텍스트 파일 생성
        report_text = f"""=== 다중 모달리티 헬스케어 AI 진단 리포트 ===
        [환자 입력 정보]
        - 연령: {age}세 | 성별: {'남성' if sex=='M' else '여성'}
        - 혈압: {resting_bp} mmHg | 콜레스테롤: {cholesterol} mg/dl

        [AI 예측 결과]
        - 심혈관 질환 위험도: {prediction_proba * 100:.1f}%

        [AI 심층 소견]
        - 가장 주의할 요인: {risk_msg}
        - 긍정적 건강 요인: {healthy_msg}

        * 본 리포트는 AI 통계 모델을 기반으로 작성되었습니다.
        ============================================="""

        # 다운로드 버튼 생성
        st.download_button(
            label="📥 내 진단 리포트 다운로드 (.txt)",
            data=report_text,
            file_name="AI_Health_Report.txt",
            mime="text/plain"
        )

# --- 5. 미구현 질병 선택 시 ---
elif disease_category == "🩸 당뇨병 위험도 분석 (준비중)":
    st.title("🩸 당뇨병 위험도 분석 (Data Loading...)")
    st.info("현재 당뇨병 데이터 파이프라인 구축 및 학습을 진행 중입니다. 다음 세션에 활성화됩니다.")
    # 멘티들이 여기에 당뇨병 코드를 채워넣게 됩니다.

elif disease_category == "🧠 뇌졸중 조기 경보 (준비중)":
    st.title("🧠 뇌졸중 조기 경보 (Data Loading...)")
    st.info("현재 뇌졸중 불균형 데이터(SMOTE) 최적화를 진행 중입니다. 다음 세션에 활성화됩니다.")
   
    # python -m streamlit run health_dashboard.py