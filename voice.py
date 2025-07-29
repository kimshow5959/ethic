import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf # 불필요한 tensorflow import 제거
# from tensorflow.keras import layers, models # 불필요한 tensorflow import 제거
import io
import soundfile as sf
import base64
from scipy.fft import fft
from scipy.signal import spectrogram

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

# MFCC 특성 추출 함수 (scipy + numpy 사용)
def extract_mfcc(audio, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
    # 음성에서 퓨리에 변환 수행
    freqs, times, Sxx = spectrogram(audio, fs=sr, nperseg=n_fft, noverlap=hop_length)
    
    # Mel 필터 뱅크 생성 (MFCC에서 사용하는 Mel 축을 변환)
    mel_filters = np.zeros((n_mels, len(freqs)))
    mel_freqs = np.linspace(0, sr / 2, n_mels + 2)
    for i in range(1, len(mel_freqs) - 1):
        start = int(np.floor(mel_freqs[i - 1] / (sr / 2) * len(freqs)))
        end = int(np.floor(mel_freqs[i] / (sr / 2) * len(freqs)))
        mel_filters[i - 1, start:end] = np.linspace(0, 1, end - start)
    
    # Mel 스펙트로그램 계산
    mel_spectrogram = np.dot(mel_filters, np.abs(Sxx))
    mel_log = np.log(mel_spectrogram + 1e-9)
    
    # MFCC 계산 (DCT 사용)
    mfcc = np.fft.dct(mel_log, type=2, axis=0)[:n_mfcc]
    return np.mean(mfcc.T, axis=0)

# 스펙트로그램 이미지 추출 함수
def extract_spectrogram(audio, sr, n_mels=128, hop_length=512):
    _, _, Sxx = spectrogram(audio, fs=sr, nperseg=2048, noverlap=hop_length)
    S_dB = 10 * np.log10(Sxx + 1e-9) # log10(0) 방지
    return S_dB

# 오디오 재생 플레이어 생성 함수
def get_audio_player(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    audio_html = f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    return audio_html

# 간단한 CNN 모델 구성 (현재 사용되지 않으므로 주석 처리 또는 삭제 가능)
# def build_cnn_model(input_shape=(128, 128, 1)):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

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
            audio, sr = sf.read(uploaded_file)
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
        S_dB = extract_spectrogram(st.session_state['audio'], st.session_state['sr'])
        ax.imshow(S_dB, aspect='auto', cmap='inferno', origin='lower')
        ax.set(title='Mel 스펙트로그램')
        st.pyplot(fig)

    # 3단계: AI 학습 및 분류
    st.subheader("3단계: AI로 분류하기")
    if st.button("학습 후 분류 실행"):
        if 'audio' not in st.session_state:
            st.warning("먼저 음성을 생성하거나 업로드해주세요.")
            return

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
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # 재현성을 위해 random_state 추가
        rf_model.fit(X_rf, y_rf)

        # 음성 데이터 처리
        mfcc = extract_mfcc(st.session_state['audio'], st.session_state['sr'])
        pred_rf = rf_model.predict([mfcc])

        st.write(f"**AI 예측 결과:** {'진짜' if pred_rf[0] == 1 else '가짜'} 음성")
        if st.session_state['is_real'] is not None:
            st.write(f"**실제 음성 여부:** {'진짜' if st.session_state['is_real'] else '가짜'} 음성")
            if (pred_rf[0] == 1 and st.session_state['is_real']) or \
               (pred_rf[0] == 0 and not st.session_state['is_real']):
                st.success("🎉 AI가 정확하게 예측했습니다!")
            else:
                st.error("🤔 AI가 잘못 예측했습니다. 더 많은 데이터와 복잡한 모델이 필요할 수 있습니다.")


# 메뉴 처리
if menu == "홈":
    st.title('Ethic is good for us')
    content_col, tips_col = st.columns([4, 1])

    with content_col:
        st.subheader("AI Ethics and Responsibility")
        st.video(url)
        st.write("""AI는 현대 사회를 변화시키는 핵심 기술입니다. 
        그러나 AI의 사용에는 윤리적 고려가 반드시 따라야 하며, 우리는 다음과 같은 원칙을 따라야 합니다:
        - **공정성 (Fairness)** - **책임성 (Accountability)** - **투명성 (Transparency)** - **프라이버시 보호 (Privacy Protection)**""")
        
        # 의견 제출 폼
        st.markdown("---")
        st.subheader("여러분의 의견을 남겨주세요")
        st.info("⚠️ **참고:** Streamlit Cloud는 기본적으로 파일 시스템이 임시적입니다. 여기에 제출된 의견은 앱이 재시작되거나 재배포될 때 사라질 수 있습니다. 영구적인 데이터 저장을 위해서는 데이터베이스 연동이 필요합니다.")
        user_opinion = st.text_area("의견을 입력하세요:")
        if st.button("의견 제출"):
            if user_opinion:
                with open("data.txt", "a", encoding='utf-8') as file: # 한글 인코딩 추가
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
    st.title("AI 윤리 개요")
    st.markdown("""
    AI 윤리는 인공지능 기술의 개발, 배포 및 사용에 있어 발생할 수 있는 윤리적, 사회적 문제를 다루는 분야입니다.
    AI가 사회에 미치는 영향이 커짐에 따라, 기술의 발전과 함께 윤리적 고려가 필수적입니다.

    ### 주요 AI 윤리 원칙
    * **공정성 (Fairness)**: AI 시스템은 특정 집단에 대한 편향이나 차별을 유발해서는 안 됩니다. 데이터 수집, 모델 학습, 결과 해석 등 모든 단계에서 공정성을 확보해야 합니다.
    * **책임성 (Accountability)**: AI 시스템의 오작동이나 잘못된 결정으로 인해 발생한 문제에 대해 누가 책임을 질 것인지 명확히 해야 합니다. 개발자, 배포자, 사용자 등 관련 주체들의 책임 범위를 설정하는 것이 중요합니다.
    * **투명성 (Transparency)**: AI 시스템이 어떻게 결정을 내리는지 이해할 수 있도록 그 작동 원리를 명확히 공개해야 합니다. 특히 중요한 결정에 사용되는 AI의 경우, '블랙박스' 문제를 해결하고 설명 가능성을 높여야 합니다.
    * **프라이버시 보호 (Privacy Protection)**: AI는 방대한 데이터를 처리하며, 이 과정에서 개인 정보가 침해될 위험이 있습니다. 데이터 수집, 저장, 사용 및 공유에 있어 개인 정보 보호 법규를 준수하고, 사용자의 동의를 얻는 등 프라이버시를 최우선으로 보호해야 합니다.
    * **안전성 (Safety)**: AI 시스템은 예측 불가능한 위험을 초래하지 않도록 안전하게 설계되고 배포되어야 합니다. 특히 자율 주행, 의료 AI 등 생명과 직결된 분야에서는 안전성 확보가 매우 중요합니다.
    * **인간 존중 (Human Dignity)**: AI는 인간의 자율성과 존엄성을 침해하지 않고, 인간의 삶을 보조하고 향상시키는 방향으로 개발되어야 합니다. 인간의 통제권을 유지하고, AI가 인간을 대체하기보다는 협력하는 관계를 지향해야 합니다.

    이러한 원칙들은 AI 기술이 사회에 긍정적인 영향을 미치고, 잠재적인 위험을 최소화하며 지속 가능한 발전을 이루는 데 중요한 지침이 됩니다.
    """)
elif menu == "딥페이크 음성":
    run_deepfake_detection()
elif menu == "참고 자료":
    st.title("참고 자료")
    st.markdown("""
    이 섹션에서는 AI 윤리 및 딥페이크 기술과 관련된 유용한 자료들을 제공합니다.

    ### AI 윤리 관련 자료
    * **AI 윤리 가이드라인**: 각국 정부나 국제기구에서 발표하는 AI 윤리 가이드라인 문서를 참고하세요.
        * [OECD AI 원칙](https://www.oecd.org/going-digital/ai/principles/) (영문)
        * [대한민국 인공지능 윤리 기준](https://www.msit.go.kr/web/msitContents/contentsView.do?cateId=MSIT_02010100&artId=1354519)
    * **학술 논문 및 보고서**: AI 윤리 분야의 최신 연구 동향을 파악할 수 있습니다.
        * Google AI Ethics: [https://ai.google/responsibility/](https://ai.google/responsibility/)
        * Microsoft AI Ethics: [https://www.microsoft.com/en-us/ai/our-approach-to-ai-ethics](https://www.microsoft.com/en-us/ai/our-approach-to-ai-ethics)
    * **온라인 강좌**: Coursera, edX 등에서 제공하는 AI 윤리 관련 강좌를 수강해 보세요.

    ### 딥페이크 기술 관련 자료
    * **뉴스 기사 및 분석 보고서**: 딥페이크 기술의 발전과 사회적 영향에 대한 최신 정보를 얻을 수 있습니다.
        * [딥페이크 기술의 현황과 미래](https://www.korea.kr/news/policyNewsView.do?newsId=148896561) (대한민국 정책브리핑)
    * **기술 블로그**: 딥페이크 생성 및 탐지 기술에 대한 심층적인 설명을 제공합니다.
    * **YouTube 영상**: 딥페이크의 원리나 실제 사례를 시각적으로 이해하는 데 도움이 됩니다.

    ### 데이터 과학 및 머신러닝 기초
    * **Numpy 공식 문서**: [https://numpy.org/doc/](https://numpy.org/doc/)
    * **Scipy 공식 문서**: [https://docs.scipy.org/doc/](https://docs.scipy.org/doc/)
    * **Scikit-learn 공식 문서**: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
    * **Streamlit 공식 문서**: [https://docs.streamlit.io/](https://docs.streamlit.io/)
    """)
    