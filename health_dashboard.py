# [T-SUM_2026-1_Proj] AI 심혈관 질환 예측 시스템 대시보드 (Streamlit)
# 2026.05.21. 최종 수정

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings, os
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="통합 헬스케어 AI 포털",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 크로스 플랫폼 정적 폰트 번들링 로직 ---
@st.cache_resource
def load_custom_font():
    font_path = "NanumGothic.ttf" # 저장한 파일명과 정확히 일치해야 합니다.
    
    if not os.path.exists(font_path):
        st.error(f"🚨 폰트 파일을 찾을 수 없습니다: {font_path}")
        return None
        
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    
    return font_name

# 함수 실행 (메모리에 한 번만 적재)
load_custom_font()

# --- 다중 모델 로드 및 학습 함수 (캐싱) ---
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

@st.cache_resource
def load_diabetes_model():
    model_path = 'diabetes_model.pkl'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

@st.cache_resource
def load_stroke_model():
    """CSV로 RandomForest 모델 학습 후 반환"""
    # CSV 경로 탐색
    candidates = [
        'healthcare-dataset-stroke-data.csv',
        os.path.join(os.path.dirname(__file__), 'healthcare-dataset-stroke-data.csv'),
    ]
    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, na_values=['N/A', 'Unknown'])
            break

    if df is None:
        return None, None, None

    df = df[df['gender'] != 'Other']
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['smoking_status'] = df['smoking_status'].fillna(df['smoking_status'].mode()[0])

    df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2})
    df['ever_married']   = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})
    df['gender']         = df['gender'].map({'Female': 0, 'Male': 1})

    df['Age_over_55']          = (df['age'] >= 55).astype(int)
    df['Obesity']              = (df['bmi'] >= 30).astype(int)
    df['Diabetic']             = (df['avg_glucose_level'] >= 126).astype(int)
    df['age_x_hypertension']   = df['age'] * df['hypertension']
    df['age_x_heart_disease']  = df['age'] * df['heart_disease']
    df['glucose_x_age']        = df['avg_glucose_level'] * df['age'] / 1000
    df['risk_score'] = (
        df['hypertension'] + df['heart_disease'] +
        df['Age_over_55'] + df['Obesity'] + df['Diabetic'] +
        (df['smoking_status'] == 2).astype(int)
    )

    X = df.drop(['id', 'stroke'], axis=1)
    y = df['stroke']
    X_enc = pd.get_dummies(X, columns=['work_type'], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=500, max_depth=10,
        class_weight='balanced_subsample',
        min_samples_leaf=2, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, X_enc.columns.tolist(), round(auc, 4)

THRESHOLD_HIGH   = 0.30
THRESHOLD_MEDIUM = 0.15

# 심장병 모델 자원 로드
heart_model, heart_scaler, heart_cols, heart_df = load_heart_model()

# 당뇨병 모델 로드 실행 
diabetes_model, diabetes_scaler = load_diabetes_model()

# 뇌졸중 모델 로드 실행
stroke_model, stroke_cols, stroke_auc = load_stroke_model()

# --- 메뉴 상수 정의 ---
MENU_HEART = "🫀 심혈관 질환 예측"
MENU_DIABETES = "🩸 당뇨병 위험도 분석"
MENU_STROKE = "🧠 뇌졸중 조기 경보 (준비중)"

st.sidebar.title("🏥 종합 진단 센터")
# 상수가 저장된 리스트를 라디오 버튼에 주입
disease_category = st.sidebar.radio("진단 과목 선택", [MENU_HEART, MENU_DIABETES, MENU_STROKE])

st.sidebar.divider()

# --- 심혈관 질환 선택 시 ---
if disease_category == MENU_HEART:
    st.sidebar.header("📋 나의 건강 지표 입력")
    
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
    st.title("🫀 AI 기반 심혈관 질환 위험도 분석 예측 시스템")
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
            st.success("✅ **안전 (Low Risk)**: 심혈관 질환 발병 가능성이 낮습니다.")
        st.metric(label="AI 예측 심혈관 질환 양성 확률", value=f"{prediction_proba * 100:.1f}%")
        st.progress(prediction_proba)
        st.divider()
        st.caption("""
        :red[**⚠️ 주의사항**]\n
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
            **좌측의 숫자 (예: 1.151)**
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
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 8px; border-left: 6px solid #1f77b4; margin: 10px 0;">
            <strong style="font-size: 16px; color: #0f172a; display: block; margin-bottom: 12px;">📢 AI 종합 진단 소견</strong>
            <ul style="padding-left: 20px; margin: 0; list-style-type: disc;">
                <li style="margin-bottom: 12px; color: #334155;">
                    <span style='color:#ff4b4b; font-weight:bold;'>주의가 필요한 요인</span>: 
                    현재 입력하신 정보 중 <span style='color:#ff4b4b; font-weight:bold;'>'{risk_feature_name} : {risk_msg}'</span> 항목이 당신의 심장 건강 위험을 높이는 가장 큰 원인입니다.
                </li>
                <li style="color: #334155;">
                    <span style='color:#00cc96; font-weight:bold;'>긍정적인 건강 요인</span>: 
                    반면 <span style='color:#00cc96; font-weight:bold;'>'{healthy_feature_name} : {healthy_msg}'</span> 항목은 현재 당신의 심장 건강을 유지하는 데 가장 긍정적인 기여를 하고 있습니다.
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
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

# --- 5. 당뇨병 선택 시 ---
elif disease_category == MENU_DIABETES:
    st.title("🩸 AI 기반 당뇨병 발병 위험도 예측 및 시뮬레이터")
    
    if diabetes_model is None or diabetes_scaler is None:
        st.error("❌ 모델 또는 스케일러 파일('diabetes_model.pkl', 'scaler.pkl')을 찾을 수 없습니다. 파일 위치를 확인하세요.")
    else:
        st.markdown("현재 **[당뇨병 부문]** 진단 모듈이 가동 중입니다. 좌측의 임상 데이터를 변경하여 실시간 예측을 확인하세요.")
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

        # --- 사이드바 UI 이식 ---
        st.sidebar.header("📋 당뇨 위험인자 입력")
        
        gen_hlth = st.sidebar.slider("평소 건강 상태 (1:최고 ~ 5:최악)", 1, 5, 3, help="주관적인 본인의 건강상태\n\n1: 아주 좋음\n\n5: 아주 나쁨")
        high_bp = st.sidebar.selectbox("고혈압 진단 여부", [0, 1], format_func=lambda x: "예" if x==1 else "아니오", help="고혈압 여부\n\n0: 없음\n\n1: 고혈압 진단받음")
        high_chol = st.sidebar.selectbox("고콜레스테롤 여부", [0, 1], format_func=lambda x: "예" if x==1 else "아니오", help="고콜레스테롤 여부\n\n0: 없음\n\n1: 고콜레스테롤 진단받음")
        age = st.sidebar.number_input("연령대 코드 (1~13)", 1, 13, 5, help="연령대 코드\n\n1: 18-24세   2: 25-29세  3: 30-34세  4: 35-39세  5: 40-44세\n\n6: 45-49세    7: 50-54세  8: 55-59세  9: 60-64세  10: 65-69세\n\n11: 70-74세 12: 75-79세 13: 80세 이상")
        bmi = st.sidebar.number_input("BMI 지수", 10.0, 60.0, 25.0, step=0.1, help="체질량지수(BMI) = 체중(kg) / 키(m)²")
        phys_act = st.sidebar.selectbox("한 달간 운동 여부", [0, 1], format_func=lambda x: "예" if x==1 else "아니오", help="한 달간 신체 활동 여부\n\n0: 운동 안함\n\n1: 운동 함")

        # --- 데이터 전처리 및 예측 ---
        input_data = [[gen_hlth, high_bp, age, high_chol, bmi, phys_act]]
        input_df = pd.DataFrame(input_data, columns=['GenHlth', 'HighBP', 'Age', 'HighChol', 'BMI', 'PhysActivity'])
        
        input_scaled = diabetes_scaler.transform(input_df)
        prob = diabetes_model.predict_proba(input_scaled)[0][1] * 100

        # --- 메인 화면 결과 출력 ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("진단 결과")
            if prob >= 50:
                st.error("🚨 **위험 (High Risk)**: 당뇨 발병 가능성이 매우 높습니다. 전문의 상담이 필요합니다.")
            elif prob >= 30:
                st.warning("⚠️ **주의 (Moderate Risk)**: 당뇨 발병 전단계일 가능성이 있습니다. 관리가 필요합니다.")
            else:
                st.success("✨ **안전 (Low Risk)**: 현재 혈당 관련 지표가 안정적입니다.")
            
            st.metric(label="AI 예측 당뇨 발병 확률", value=f"{prob:.1f}%")
            st.progress(prob / 100.0)

        # --- 혁신 기능: What-If 시뮬레이터 ---
        with col2:
            st.subheader("💡 라이프스타일 개선 시뮬레이션")
            st.info("현재 상태에서 통제 가능한 요인(운동, 체중)을 개선했을 때의 효과를 계산합니다.")
            
            # 시나리오: 운동을 안하던 사람이 운동을 시작하고, BMI를 2.0 감량했을 때
            simulated_bmi = max(10.0, bmi - 2.0) # BMI가 10 이하로 떨어지지 않게 방어 로직
            simulated_act = 1 # 무조건 운동을 한다고 가정
            
            sim_df = pd.DataFrame([[gen_hlth, high_bp, age, high_chol, simulated_bmi, simulated_act]], 
                                  columns=['GenHlth', 'HighBP', 'Age', 'HighChol', 'BMI', 'PhysActivity'])
            sim_scaled = diabetes_scaler.transform(sim_df)
            sim_prob = diabetes_model.predict_proba(sim_scaled)[0][1] * 100
            
            prob_diff = prob - sim_prob
            
            if prob_diff > 0:
                st.metric(label="개선 후 예상 당뇨 발병 확률", value=f"{sim_prob:.1f}%", delta=f"-{prob_diff:.1f}% 감소", delta_color="inverse")
                st.markdown(f"**Action Plan:** 한 달간 꾸준히 운동을 시작하고 체중을 조절하여 BMI를 2.0 낮추면, 당뇨 발병 위험을 **{prob_diff:.1f}%p** 낮출 수 있습니다.")
            else:
                st.metric(label="개선 후 예상 당뇨 발병 확률", value=f"{sim_prob:.1f}%", delta="추가 감소 여력 낮음", delta_color="off")
                st.markdown("**Action Plan:** 현재 이미 훌륭한 생활 습관을 유지하고 있습니다. 지금의 운동량과 체중을 유지하세요.")
        
    st.divider()
    st.subheader("📊 환자 건강 지표 상대적 포지셔닝 (BMI 기준)")
    
    # 원본 데이터를 메모리에 캐싱하여 로드 속도 최적화
    @st.cache_data
    def load_visual_data():
        # 멘티가 업로드한 데이터셋 활용
        return pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2021.csv')
    
    try:
        df_vis = load_visual_data()
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.font_manager as fm # 상단에 없다면 추가
        
        # 폰트 파일 경로 정의 및 폰트 객체 생성
        font_path = "NanumGothic.ttf"
        fe = fm.FontEntry(fname=font_path, name='NanumGothic')
        fm.fontManager.ttflist.insert(0, fe) # 폰트 매니저 헤드에 강제 인서트
        
        # 명시적으로 사용할 폰트 프로퍼티 정의 (C언어의 구조체 정의와 유사)
        korean_font = fm.FontProperties(fname=font_path)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # 분포도 그리기
        sns.kdeplot(data=df_vis[df_vis['Diabetes_binary'] == 0], x='BMI', fill=True, color="cornflowerblue", label="정상 그룹 밀집도 (Safe)", ax=ax)
        sns.kdeplot(data=df_vis[df_vis['Diabetes_binary'] == 1], x='BMI', fill=True, color="indianred", label="당뇨 그룹 밀집도 (Risk)", ax=ax)
        
        # 환자 마커
        ax.axvline(bmi, color='black', linestyle='--', linewidth=2.5, label=f"현재 환자 위치 (BMI: {bmi:.1f})")
        if 'simulated_bmi' in locals() and simulated_bmi < bmi:
            ax.axvline(simulated_bmi, color='green', linestyle='-.', linewidth=2.5, label=f"행동 개선 후 목표 위치 (BMI: {simulated_bmi:.1f})")
            
        # 🔥 [핵심 변경 포인트] fontproperties 인자를 사용하여 시스템 캐시를 우회하고 직접 주입
        ax.set_title("전체 환자 통계 대비 나의 체질량지수(BMI) 위치 분석", fontproperties=korean_font, fontsize=14, fontweight='bold')
        ax.set_xlabel("BMI 지수", fontproperties=korean_font, fontsize=12)
        ax.set_ylabel("데이터 밀집도 (Density)", fontproperties=korean_font, fontsize=12)
        
        # 범례(Legend) 한글 깨짐 방지 처리
        ax.legend(prop=korean_font, loc='upper right')
        
        ax.set_xlim(10, 60)
        ax.set_yticks([]) 
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("⚠️ 시각화용 원본 데이터셋 파일을 찾을 수 없습니다.")

# --- 뇌졸중 선택 시 ---
elif disease_category == MENU_STROKE:
    def get_risk_level(prob: float):
        if prob >= THRESHOLD_HIGH:
            return "고위험", "#e53e3e", "🔴"
        elif prob >= THRESHOLD_MEDIUM:
            return "중위험", "#dd6b20", "🟠"
        else:
            return "저위험", "#38a169", "🟢"

    def get_risk_class(prob: float):
        if prob >= THRESHOLD_HIGH:   return "risk-high"
        elif prob >= THRESHOLD_MEDIUM: return "risk-medium"
        return "risk-low"

    def get_recommendations(inputs: dict, prob: float) -> list[dict]:
        """
        각 위험인자 분석 후 우선순위 권고사항 반환
        반환: [{'priority': '즉시', 'icon': '🚨', 'message': '...', 'detail': '...'}, ...]
        """
        recs = []

        # 혈압
        if inputs['hypertension']:
            recs.append({
                'priority': '즉시', 'color': 'red',
                'icon': '💊',
                'message': '고혈압 관리 강화',
                'detail': '혈압 목표: 수축기 130mmHg 미만. 저염식(하루 소금 5g 이하), 규칙적 복약, 주 2회 이상 혈압 측정 필요.'
            })

        # 심장질환
        if inputs['heart_disease']:
            recs.append({
                'priority': '즉시', 'color': 'red',
                'icon': '❤️‍🩹',
                'message': '심장 전문의 정기 추적',
                'detail': '6개월마다 심전도·심초음파 검사 권장. 항응고제 또는 항혈소판제 복용 여부를 의사와 상의하세요.'
            })

        # 혈당
        if inputs['avg_glucose_level'] >= 126:
            recs.append({
                'priority': '즉시', 'color': 'red',
                'icon': '🩸',
                'message': '혈당 조절 필요 (당뇨 의심)',
                'detail': f'현재 혈당 {inputs["avg_glucose_level"]:.0f}mg/dL — 당화혈색소(HbA1c) 검사 권장. 탄수화물 섭취 제한, 식후 혈당 모니터링.'
            })
        elif inputs['avg_glucose_level'] >= 100:
            recs.append({
                'priority': '주의', 'color': 'orange',
                'icon': '🩸',
                'message': '공복혈당 장애 (전당뇨 구간)',
                'detail': f'현재 혈당 {inputs["avg_glucose_level"]:.0f}mg/dL — 정상 범위(70-99)를 초과. 정제 탄수화물·당류 섭취 줄이기, 연 1회 당뇨 검사 권장.'
            })

        # 비만
        if inputs['bmi'] >= 30:
            recs.append({
                'priority': '주의', 'color': 'orange',
                'icon': '⚖️',
                'message': '체중 감량 권고 (비만)',
                'detail': f'현재 BMI {inputs["bmi"]:.1f} — 5~10% 체중 감량만으로도 뇌졸중 위험 유의미하게 감소. 주 150분 이상 중강도 운동 권장.'
            })
        elif inputs['bmi'] >= 25:
            recs.append({
                'priority': '관찰', 'color': 'orange',
                'icon': '⚖️',
                'message': '과체중 — 체중 유지 권고',
                'detail': f'현재 BMI {inputs["bmi"]:.1f} — 정상 범위(18.5-24.9) 초과. 식이 조절과 규칙적 운동으로 체중 유지.'
            })

        # 흡연
        if inputs['smoking_status'] == 2:
            recs.append({
                'priority': '즉시', 'color': 'red',
                'icon': '🚭',
                'message': '즉시 금연 필요',
                'detail': '흡연은 뇌졸중 위험을 2배 이상 증가. 금연 후 5년 내 위험도 비흡연자 수준으로 회복. 금연 클리닉·니코틴 대체요법 활용 권장.'
            })
        elif inputs['smoking_status'] == 1:
            recs.append({
                'priority': '관찰', 'color': 'orange',
                'icon': '🚭',
                'message': '과거 흡연 이력 — 지속 금연 유지',
                'detail': '금연 유지 중이라면 계속 이어가세요. 혈관 기능은 금연 후에도 서서히 회복됩니다.'
            })

        # 나이
        if inputs['age'] >= 75:
            recs.append({
                'priority': '주의', 'color': 'orange',
                'icon': '🏥',
                'message': '고령 — 연 1회 뇌혈관 정밀 검사 권고',
                'detail': '75세 이상은 뇌졸중 위험이 급격히 증가. MRI·MRA 뇌혈관 검사, 경동맥 초음파 검사 권장.'
            })
        elif inputs['age'] >= 55:
            recs.append({
                'priority': '관찰', 'color': 'orange',
                'icon': '🏥',
                'message': '55세 이상 — 정기 뇌혈관 검진 권고',
                'detail': '2년마다 뇌혈관 건강 체크. 혈압·혈당·콜레스테롤 동시 관리.'
            })

        # 저위험 + 개선사항 없을 때
        if not recs and prob < THRESHOLD_MEDIUM:
            recs.append({
                'priority': '양호', 'color': 'green',
                'icon': '✅',
                'message': '현재 위험인자 없음 — 건강 유지 권고',
                'detail': '규칙적 운동, 균형 잡힌 식단, 금연·절주를 유지하면 뇌졸중 위험을 최소화할 수 있습니다. 연 1회 기본 건강검진 권장.'
            })

        # 우선순위 정렬
        priority_order = {'즉시': 0, '주의': 1, '관찰': 2, '양호': 3}
        recs.sort(key=lambda x: priority_order.get(x['priority'], 9))
        return recs

    def predict(model, feature_cols: list, inputs: dict) -> float:
        smoking_num  = inputs['smoking_status']
        married_num  = 1 if inputs['ever_married'] == 'Yes' else 0
        residence_num = 1 if inputs['Residence_type'] == 'Urban' else 0
        gender_num   = 1 if inputs['gender'] == 'Male' else 0

        row = {
            'gender':              gender_num,
            'age':                 inputs['age'],
            'hypertension':        int(inputs['hypertension']),
            'heart_disease':       int(inputs['heart_disease']),
            'ever_married':        married_num,
            'Residence_type':      residence_num,
            'avg_glucose_level':   inputs['avg_glucose_level'],
            'bmi':                 inputs['bmi'],
            'smoking_status':      smoking_num,
            'Age_over_55':         int(inputs['age'] >= 55),
            'Obesity':             int(inputs['bmi'] >= 30),
            'Diabetic':            int(inputs['avg_glucose_level'] >= 126),
            'age_x_hypertension':  inputs['age'] * int(inputs['hypertension']),
            'age_x_heart_disease': inputs['age'] * int(inputs['heart_disease']),
            'glucose_x_age':       inputs['avg_glucose_level'] * inputs['age'] / 1000,
            'risk_score': (
                int(inputs['hypertension']) + int(inputs['heart_disease']) +
                int(inputs['age'] >= 55) + int(inputs['bmi'] >= 30) +
                int(inputs['avg_glucose_level'] >= 126) + int(smoking_num == 2)
            ),
            # work_type 원-핫
            'work_type_Govt_job':      int(inputs['work_type'] == 'Govt_job'),
            'work_type_Never_worked':  int(inputs['work_type'] == 'Never_worked'),
            'work_type_Private':       int(inputs['work_type'] == 'Private'),
            'work_type_Self-employed': int(inputs['work_type'] == 'Self-employed'),
            'work_type_children':      int(inputs['work_type'] == 'children'),
        }

        df_input = pd.DataFrame([row])
        # 모델 학습 시 사용된 컬럼과 정렬 맞춤
        for c in feature_cols:
            if c not in df_input.columns:
                df_input[c] = 0
        df_input = df_input[feature_cols]

        prob = model.predict_proba(df_input)[0][1]
        return float(prob)

    def draw_gauge(prob: float):
        import matplotlib.font_manager as fm

        # 한글 폰트 직접 탐색
        _korean_fonts = [
            'Malgun Gothic', 'AppleGothic', 'NanumGothic', 'NanumBarunGothic',
            'UnDotum', 'Noto Sans CJK KR', 'Noto Sans KR', 'DejaVu Sans'
        ]
        _available = {f.name for f in fm.fontManager.ttflist}
        _font = next((f for f in _korean_fonts if f in _available), 'DejaVu Sans')

        fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect='equal'))
        fig.patch.set_alpha(0)
        ax.set_axis_off()

        # 호 배경 (회색)
        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), lw=22, color='#e2e8f0', solid_capstyle='butt', transform=ax.transData)

        # 호 채우기
        fill_theta = np.linspace(np.pi, np.pi - np.pi * prob, 200)
        color = '#e53e3e' if prob >= THRESHOLD_HIGH else '#dd6b20' if prob >= THRESHOLD_MEDIUM else '#38a169'
        ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=22, color=color, solid_capstyle='butt')

        # 중앙 텍스트
        fp = fm.FontProperties(family=_font)
        ax.text(0, -0.05, f"{prob*100:.1f}%", ha='center', va='center',
                fontsize=26, fontweight='bold', color=color, fontproperties=fp)
        ax.text(0, -0.42, '뇌졸중 위험 확률', ha='center', va='center',
                fontsize=9, color='#718096', fontproperties=fp)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.1)
        return fig

    def main():
        # 헤더
        st.markdown("## 🧠 뇌졸중 위험도 예측")
        st.markdown("개인 건강 정보를 입력하면 AI가 뇌졸중 위험도를 분석하고 맞춤 개선사항을 제공합니다.")
        st.markdown("---")

        # 모델 로드
        stroke_model, stroke_cols, stroke_auc = load_stroke_model()
        if stroke_model is None:
            st.error("⚠️ `healthcare-dataset-stroke-data.csv` 파일을 앱과 같은 폴더에 넣어주세요.")
            st.stop()

        st.caption(f"모델 성능 — ROC-AUC: **{stroke_auc}** | 학습 데이터: 5,109명 | 위험 임계값: ≥30% 고위험")

        # ── 입력 폼 ──────────────────────────────────────────
        with st.form("input_form"):
            st.markdown('<div class="section-title">📋 건강 정보 입력</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**기본 정보**")
                gender    = st.selectbox("성별",      ["여성", "남성"], index=1)
                age       = st.slider("나이", 1, 82, 45)
                married   = st.selectbox("결혼 여부", ["미혼", "기혼"], index=1)
                residence = st.selectbox("거주 지역", ["시골", "도시"], index=1)
                work_type = st.selectbox("직업 유형", ["회사원", "자영업/프리랜서", "공무원", "학생", "무직"])

            with col2:
                st.markdown("**건강 수치**")
                glucose = st.number_input("평균 혈당 수치 (mg/dL)", 50.0, 300.0, 90.0, step=1.0,
                                        help="공복 혈당 기준: 정상 70-99 / 전당뇨 100-125 / 당뇨 ≥126")
                bmi     = st.number_input("BMI (체질량 지수)", 10.0, 60.0, 24.0, step=0.1,
                                        help="정상 18.5-24.9 / 과체중 25-29.9 / 비만 ≥30")

                st.markdown("")
                st.markdown("**BMI 참고 기준**")
                st.markdown("🟢 정상: 18.5–24.9 &nbsp; 🟠 과체중: 25–29.9 &nbsp; 🔴 비만: ≥30",
                            unsafe_allow_html=True)

            with col3:
                st.markdown("**기저질환 및 생활습관**")
                hypertension  = st.checkbox("고혈압 진단 이력", value=False)
                heart_disease = st.checkbox("심장질환 진단 이력", value=False)
                smoking = st.selectbox("흡연 상태", [
                    "never smoked", "formerly smoked", "smokes"
                ], format_func=lambda x: {
                    "never smoked": "비흡연",
                    "formerly smoked": "과거 흡연 (현재 금연)",
                    "smokes": "현재 흡연 중"
                }[x])

                st.markdown("")
                st.info("💡 모든 정보는 예측에만 사용되며 저장되지 않습니다.")

            submitted = st.form_submit_button("🔍 위험도 분석하기", use_container_width=True, type="primary")

        # ── 결과 출력 ─────────────────────────────────────────
        if submitted:
            gender_map    = {"여성": "Female",       "남성": "Male"}
            married_map   = {"미혼": "No",           "기혼": "Yes"}
            residence_map = {"시골": "Rural",        "도시": "Urban"}
            work_map      = {
            "회사원":          "Private",
            "자영업/프리랜서": "Self-employed",
            "공무원":          "Govt_job",
            "학생":            "children",
            "무직":            "Never_worked",
            }
            smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2}
            
            inputs = {
            'gender':            gender_map[gender],
            'age':               age,
            'ever_married':      married_map[married],
            'Residence_type':    residence_map[residence],
            'work_type':         work_map[work_type],
            'avg_glucose_level': glucose,
            'bmi':               bmi,
            'hypertension':      hypertension,
            'heart_disease':     heart_disease,
            'smoking_status':    smoking_map[smoking],
            }

            prob       = predict(stroke_model, stroke_cols, inputs)
            level, color, emoji = get_risk_level(prob)
            risk_class = get_risk_class(prob)
            recs       = get_recommendations(inputs, prob)

            st.markdown("---")
            st.markdown("### 📊 분석 결과")

            res_col1, res_col2 = st.columns([1, 2])

            # 왼쪽: 게이지 + 등급
            with res_col1:
                st.pyplot(draw_gauge(prob), use_container_width=True)
                plt.close()

                badge_color_map = {'고위험': 'risk-high', '중위험': 'risk-medium', '저위험': 'risk-low'}
                st.markdown(f"""
                <div class="risk-card {risk_class}" style="text-align:center; padding:16px;">
                    <div style="font-size:2.2rem;">{emoji}</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{color};">{level}</div>
                    <div style="font-size:0.9rem; color:#4a5568; margin-top:6px;">
                        {'뇌졸중 위험이 높습니다. 즉시 의료 전문가와 상담하세요.' if level == '고위험'
                        else '일부 위험인자가 있습니다. 생활습관 개선이 필요합니다.' if level == '중위험'
                        else '현재 위험도는 낮습니다. 건강한 생활을 유지하세요.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 오른쪽: 위험인자 요약 + 권고사항
            with res_col2:
                # 위험인자 현황 요약
                st.markdown('<div class="section-title">⚠️ 위험인자 현황</div>', unsafe_allow_html=True)

                factor_cols = st.columns(3)
                factors = [
                    ("나이",        f"{age}세",         age >= 55),
                    ("혈당",        f"{glucose:.0f} mg/dL", glucose >= 126),
                    ("BMI",        f"{bmi:.1f}",        bmi >= 30),
                    ("고혈압",      "있음" if hypertension else "없음",   hypertension),
                    ("심장질환",    "있음" if heart_disease else "없음",  heart_disease),
                    ("흡연",        {0: "비흡연", 1: "과거흡연", 2: "현재흡연"}[smoking_map[smoking]],
                                smoking_map[smoking] == 2),
                ]

                for i, (name, val, is_risk) in enumerate(factors):
                    with factor_cols[i % 3]:
                        icon = "🔴" if is_risk else "🟢"
                        bg   = "#fff0f0" if is_risk else "#f0fff4"
                        bc   = "#e53e3e" if is_risk else "#38a169"
                        st.markdown(f"""
                        <div style="background:{bg}; border:1.5px solid {bc}; border-radius:10px;
                                    padding:10px 12px; text-align:center; margin-bottom:8px;">
                            <div style="font-size:1.1rem;">{icon}</div>
                            <div style="font-size:0.78rem; color:#718096; font-weight:600;">{name}</div>
                            <div style="font-size:1rem; font-weight:700; color:{bc};">{val}</div>
                        </div>""", unsafe_allow_html=True)

            # 개선 권고사항
            st.markdown("---")
            st.markdown("### 💡 맞춤 개선 권고사항")

            if not recs:
                st.success("현재 뚜렷한 위험인자가 없습니다. 건강한 생활습관을 유지하세요!")
            else:
                priority_colors = {'즉시': '🔴', '주의': '🟠', '관찰': '🟡', '양호': '🟢'}
                badge_bg = {'즉시': ('#fed7d7','#c53030'), '주의': ('#feebc8','#c05621'),
                            '관찰': ('#fefcbf','#744210'), '양호': ('#c6f6d5','#276749')}

                for rec in recs:
                    bg, fc = badge_bg.get(rec['priority'], ('#eee','#333'))
                    with st.expander(f"{rec['icon']} {rec['message']}  {priority_colors.get(rec['priority'],'')} **[{rec['priority']}]**", expanded=(rec['priority']=='즉시')):
                        st.markdown(f"""
                        <div style="background:#f8fafc; border-radius:8px; padding:14px 16px;
                                    border-left:4px solid {fc};">
                            <span style="background:{bg}; color:{fc}; padding:2px 10px;
                                border-radius:12px; font-size:0.8rem; font-weight:700;">
                                {rec['priority']} 조치
                            </span>
                            <p style="margin-top:10px; color:#2d3748; line-height:1.7;">{rec['detail']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # 면책 조항
            st.markdown("---")
            st.caption(
                "⚕️ **의료 면책 조항**: 이 예측 결과는 공개 데이터셋(n=5,109)으로 학습된 AI 모델의 참고 정보로, "
                "의학적 진단을 대체하지 않습니다. 건강에 이상이 있거나 결과가 고위험으로 나온 경우 반드시 "
                "의료 전문가와 상담하세요."
            )

        # 사이드바 — 위험 기준 안내
        with st.sidebar:
            st.markdown("### 📌 위험도 판단 기준")
            st.markdown("""
    | 등급 | 확률 | 설명 |
    |------|------|------|
    | 🔴 고위험 | ≥ 30% | 즉시 전문의 상담 |
    | 🟠 중위험 | 15~30% | 생활습관 개선 필요 |
    | 🟢 저위험 | < 15% | 정기 관리 권고 |

    **임계값 근거**  
    모델 임계값 0.30 적용 시:
    - 뇌졸중 탐지율: **80%**
    - 오탐율: 23.7%
    - ROC-AUC: **0.817**

    의료 맥락에서는 뇌졸중을 놓치지 않는 것(민감도)이 최우선입니다.
            """)

            st.markdown("---")
            st.markdown("### 🩺 주요 위험인자")
            st.markdown("""
    - **나이** (55세 이상 시 위험 급증)
    - **고혈압** (수축기 ≥140mmHg)
    - **심장질환** (심방세동 포함)
    - **혈당** (≥126mg/dL = 당뇨)
    - **BMI** (≥30 = 비만)
    - **흡연** (현재 흡연)
            """)


    if __name__ == "__main__" or True:
        main()

# --- 정의되지 않은 카테고리 접근 시 (예외 처리) ---
else:
    st.error("🚨 시스템 라우팅 오류가 발생했습니다. 정의되지 않은 카테고리 접근입니다.")
    st.warning("개발자 콘솔을 확인하거나 사이드바 상수를 동기화하십시오.")
    
# python -m streamlit run health_dashboard.py