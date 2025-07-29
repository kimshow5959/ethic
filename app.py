import streamlit as st
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
st.set_page_config(page_title="딥페이크 음성 탐지", layout="wide")

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
def run():
    # 튜토리얼 설명
    st.title("🎙️ 딥페이크 음성 탐지 웹앱")
    st.header("튜토리얼: 딥페이크 음성 탐지 이해하기")
    st.markdown("""
    이 웹앱은 고등학생을 대상으로 AI가 **진짜 음성**과 **딥페이크 음성**을 어떻게 구별하는지 학습하도록 제작되었습니다. 아래 튜토리얼을 따라보세요:
    
    ### 딥페이크 음성이란?
    - **진짜 음성**: 유튜브 인터뷰, 팟캐스트, 뉴스 등에서 자연스럽게 녹음된 사람의 목소리입니다.
    - **딥페이크 음성**: AI가 사람의 목소리를 모방해 만들어낸 음성입니다.
    - **왜 중요할까요?**: 딥페이크는 가짜 뉴스나 허위 정보를 퍼뜨릴 때 이용될 수 있어 위험합니다. 이 앱은 AI가 어떻게 이를 탐지할 수 있는지 보여줍니다.
    
    ### 앱 동작 원리
    1. **오디오 생성 또는 업로드**  
       - 예시 '진짜' 또는 '가짜' 음성 생성 버튼 클릭  
       - 또는 직접 WAV 파일 업로드 가능  
    2. **오디오 분석**  
       - 음성에서 **MFCC** (음성 특징)와 **스펙트로그램** (시각적 주파수 패턴) 추출  
    3. **AI 모델 학습**  
       - **랜덤 포레스트**: MFCC 특징 기반 분류  
       - **CNN (합성곱 신경망)**: 스펙트로그램 이미지 분석  
    4. **음성 분류**  
       - AI가 음성의 진위(진짜/가짜) 및 확률을 예측해 출력  
    5. **윤리적 사고**  
       - 딥페이크 기술의 오용과 탐지의 사회적 의미를 토론하세요.
    
    ### 사용 방법
    - 음성을 생성하거나 업로드하세요.  
    - 청취하고 스펙트로그램을 확인하세요.  
    - **[학습 후 분류]** 버튼을 누르고 AI 예측 결과를 보세요.  
    - 토론 질문을 통해 AI 윤리와 책임에 대해 함께 생각해보세요.
    
    ### 참고
    실제 딥페이크 탐지에는 **LSTM, Transformer** 같은 복잡한 모델과 대용량 데이터가 사용되지만, 이 앱에서는 학생이 이해하기 쉬운 **랜덤 포레스트**와 **CNN**을 사용했습니다.
    """)

    # 메인 앱 UI
    st.header("딥페이크 음성 탐지 시스템")

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
        audio, sr = librosa.load(uploaded_file, sr=22050)
        st.session_state['audio'] = audio
        st.session_state['sr'] = sr
        st.session_state['is_real'] = None
        st.markdown("**✔️ 업로드된 음성**")
        st.markdown(get_audio_player(audio, sr), unsafe_allow_html=True)

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
            X_rf.append
