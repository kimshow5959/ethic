import streamlit as st
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import io
import soundfile as sf
import base64

# 페이지 설정
st.set_page_config(layout='wide', page_title='EthicApp')

# 사이드바 메뉴 선택
menu = st.sidebar.radio("Menu", ["홈", "AI 윤리 개요", "딥페이크 음성", "참고 자료"])

# YouTube 영상 링크
url = 'https://www.youtube.com/watch?v=XyEOEBsa8I4'

# 합성 오디오 생성 함수
def generate_synthetic_audio(is_real=True, duration=3, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    if is_real:
        # 자연스러운 주파수를 가진 진짜 음성 모사
        freq = 200 + 100 * np.sin(2 * np.pi * 0.1 * t)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
    else:
        # 인위적 패턴과 노이즈를 더한 딥페이크 음성 모사
        freq = 200 + 50 * np.sin(2 * np.pi * 0.2 * t)
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    return audio, sr

# MFCC 특성 추출 함수
def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# 스펙트로그램 이미지 추출 함수
def extract_spectrogram(audio, sr, n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = S_dB[:, :128]
    if S_dB.shape[1] < 128:
        S_dB = np.pad(S_dB, ((0, 0), (0, 128 - S_dB.shape[1])), mode='constant')
    return S_dB

# 오디오 재생 플레이어 생성 함수
def get_audio_player(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    audio_html = f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    return audio_html

# 간단한 CNN 모델 구성
def build_cnn_model(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 딥페이크 음성 탐지 앱 실행 함수
def run_deepfake_detection():
    st.title("🎙️ 딥페이크 음성 탐지 웹앱")
    st.header("튜토리얼: 딥페이크 음성 탐지 이해하기")
    st.markdown("""
    이 웹앱은 고등학생을 대상으로 AI가 **진짜 음성**과 **딥페이크 음성**을 어떻게 구별하는지 학습하도록 제작되었습니다.
    아래 튜토리얼을 따라보세요:
    
    ### 딥페이크 음성이란?
    - **진짜 음성**: 유튜브 인터뷰, 팟캐스트, 뉴스 등에서 자연스럽게 녹음된 사람의 목소리입니다.
    - **딥페이크 음성**: AI가 사람의 목소리를 모방해 만들어낸 음성입니다.
    - **왜 중요할까요?**: 딥페이크는 가짜 뉴스나 허위 정보를 퍼뜨릴 때 이용될 수 있어 위험합니다. 이 앱은 AI가 어떻게 이를 탐지할 수 있는지 보여줍니다.
    
    ### 사용 방법
    - 음성을 생성하거나 업로드하세요.
    - 청취하고 스펙트로그램을 확인하세요.
    - **[학습 후 분류]** 버튼을 누르고 AI 예측 결과를 보세요.
    - 토론 질문을 통해 AI 윤리와 책임에 대해 함께 생각해보세요.
    """)
    
    # 1단계: 음성 생성 또는 업로드
    st.subheader("1단계: 음성 생성 또는 업로드")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("진짜 음성 생성"):
            audio, sr = generate_synthetic_audio(is_real=True)
            st.session_state['audio'] = audio
            st.session_state['sr'] = sr
            st.session_state['is_real'] = True
            st.markdown("**✔️ 진짜 음성 샘플 생성됨**")
            st.markdown(get_audio_player(audio, sr), unsafe_allow_html=True)

    with col2:
        if st.button("가짜 음성 생성"):
            audio, sr = generate_synthetic_audio(is_real=False)
            st.session_state['audio'] = audio
            st.session_state['sr'] = sr
            st.session_state['is_real'] = False
            st.markdown("**✔️ 가짜 음성 샘플 생성됨**")
            st.markdown(get_audio_player(audio, sr), unsafe_allow_html=True)

    uploaded_file = st.file_uploader("또는 WAV 파일 업로드", type=["wav"])
    if uploaded_file:
        try:
            audio, sr = librosa.load(uploaded_file, sr=22050)
            st.session_state['audio'] = audio
            st.session_state['sr'] = sr
            st.session_state['is_real'] = None
            st.markdown("**✔️ 업로드된 음성**")
            st.markdown(get_audio_player(audio, sr), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"파일을 로드하는 데 실패했습니다: {e}")

    # 2단계: 스펙트로그램 시각화
    if 'audio' in st.session_state:
        st.subheader("2단계: 스펙트로그램 확인")
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=st.session_state['audio'], sr=st.session_state['sr'])
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=st.session_state['sr'], x_axis='time', y_axis='mel', ax=ax)
        ax.set(title='Mel 스펙트로그램')
        st.pyplot(fig)

    # 3단계: AI 학습 및 분류
    st.subheader("3단계: AI로 분류하기")
    if st.button("학습 후 분류 실행"):
        # 랜덤 포레스트용 데이터 생성
        X_rf, y_rf = [], []
        for _ in range(50):
            ra, sr = generate_synthetic_audio(is_real=True)
            fa, _ = generate_synthetic_audio(is_real=False)
            X_rf.append(extract_mfcc(ra, sr))
            X_rf.append(extract_mfcc(fa, sr))
            y_rf.append(1)
            y_rf.append(0)

        X_rf = np.array(X_rf)
        y_rf = np.array(y_rf)

        # 랜덤 포레스트 모델 훈련
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_rf, y_rf)

        # 음성 데이터 처리
        mfcc = extract_mfcc(st.session_state['audio'], st.session_state['sr'])
        pred_rf = rf_model.predict([mfcc])

        st.write(f"랜덤 포레스트 예측: {'진짜' if pred_rf[0] == 1 else '가짜'} 음성")

# 메뉴 처리
if menu == "홈":
    st.title('Ethic is good for us')
    content_col, tips_col = st.columns([4, 1])

    with content_col:
        st.subheader("AI Ethics and Responsibility")
        st.video(url)
        st.write("""AI는 현대 사회를 변화시키는 핵심 기술입니다.  
        그러나 AI의 사용에는 윤리적 고려가 반드시 따라야 하며, 우리는 다음과 같은 원칙을 따라야 합니다:
        - **공정성 (Fairness)**  
        - **책임성 (Accountability)**  
        - **투명성 (Transparency)**  
        - **프라이버시 보호 (Privacy Protection)**""")
        
        # 의견 제출 폼
        user_opinion = st.text_area("여러분의 의견을 남겨주세요:")
        if st.button("제출"):
            if user_opinion:
                with open("data.txt", "a") as file:
                    file.write(f"{user_opinion}\n")
                st.success("의견이 성공적으로 제출되었습니다.")
            else:
                st.warning("의견을 입력해주세요.")

    with tips_col:
        st.subheader("AI 윤리 개요")
        st.markdown("""
        1. **공정성 (Fairness)**: AI가 편향된 결정을 내리지 않도록 해야 합니다.
        2. **책임성 (Accountability)**: AI 시스템의 결정을 인간이 책임져야 합니다.
        3. **투명성 (Transparency)**: AI 시스템의 동작 원리를 명확히 해야 합니다.
        4. **프라이버시 보호 (Privacy Protection)**: AI는 사용자 데이터를 안전하게 보호해야 합니다.
        """)

elif menu == "AI 윤리 개요":
    st.write("AI 윤리 원칙에 대해 더 알고 싶다면, 위의 설명을 참고하세요.")

elif menu == "딥페이크 음성":
    run_deepfake_detection()

elif menu == "참고 자료":
    st.write("참고 자료 섹션입니다. 관련 문서 및 링크를 제공합니다.")
