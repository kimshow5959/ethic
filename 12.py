import streamlit as st
import os

# 페이지 설정
st.set_page_config(layout='wide', page_title='EthicApp')

# 사이드바 메뉴 선택
menu = st.sidebar.radio("Menu", ["홈", "AI 윤리 개요", "딥페이크 음성", "참고 자료"])

# YouTube 영상 링크
url = 'https://www.youtube.com/watch?v=XyEOEBsa8I4'

# 메뉴 처리
if menu == "홈":
    st.title('Ethic is good for us')
    content_col, tips_col = st.columns([4, 1])

    with content_col:
        st.subheader("AI Ethics and Responsibility")
        st.video(url)
        st.write("""
        인공지능(AI)은 현대 사회를 변화시키는 핵심 기술입니다.  
        그러나 AI의 사용에는 윤리적 고려가 반드시 따라야 하며, 우리는 다음과 같은 원칙을 따라야 합니다:
        - **공정성 (Fairness)**  
        - **책임성 (Accountability)**  
        - **투명성 (Transparency)**  
        - **프라이버시 보호 (Privacy)**
        """)
        st.markdown("#### ✍️ 당신의 생각을 공유해주세요")
        user_input = st.text_area("인공지능 윤리에 대한 의견 또는 질문을 작성해주세요:", height=100)
        
        # 제출 버튼 처리
        if st.button("제출하기"):
            if user_input.strip():
                # 제출된 입력을 data.txt에 저장
                try:
                    with open("data.txt", "a", encoding="utf-8") as f:
                        f.write(user_input + "\n---\n")
                    st.success("의견이 성공적으로 저장되었습니다.")
                except Exception as e:
                    st.error(f"저장 중 오류가 발생했습니다: {e}")
            else:
                st.warning("내용을 입력해주세요.")

    with tips_col:
        st.subheader("Tips...")
        st.markdown("""
        ✅ **AI 윤리 체크리스트**  
        - [ ] 데이터 편향 점검  
        - [ ] 결과 설명 가능성 확보  
        - [ ] 사용자 동의 확보  
        - [ ] 지속적 모니터링 체계  

        📌 **참고 링크**  
        - [OECD AI 원칙](https://oecd.ai/en/dashboards)  
        - [AI 윤리 가이드라인 (EU)](https://digital-strategy.ec.europa.eu/en/policies/european-approach-artificial-intelligence)
        """)

elif menu == "딥페이크 음성":
    st.title("🎙️ 딥페이크 음성 탐지")
    try:
        # app.py가 실제로 모듈로 임포트될 수 있도록 수정
        import app
        
        # app.py에서 실행할 함수 호출
        if hasattr(app, 'run'):
            app.run()  # app.py의 run() 함수 호출
        else:
            st.error("app.py에 'run' 함수가 정의되어 있지 않습니다.")
    except ImportError:
        st.error("딥페이크 음성 탐지 모듈(app.py)을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"딥페이크 음성 탐지에서 오류가 발생했습니다: {e}")

elif menu == "AI 윤리 개요":
    st.title("AI 윤리 개요")
    st.write("공정성, 책임성, 투명성, 프라이버시 등 윤리적 AI 원칙을 설명합니다.")

elif menu == "참고 자료":
    st.title("참고 자료")
    st.markdown("""
    - [OECD AI 원칙](https://oecd.ai/en/dashboards)  
    - [EU AI 윤리 가이드라인](https://digital-strategy.ec.europa.eu/en/policies/european-approach-artificial-intelligence)
    """)

# 학생 데이터 버튼
if st.sidebar.button("학생데이터(더블클릭)"):
    try:
        if not os.path.exists("data.txt"):
            st.error("data.txt 파일이 존재하지 않습니다. 데이터를 먼저 입력해 주세요.")
        else:
            with open("data.txt", "r", encoding="utf-8") as f:
                student_data = f.read()
            st.subheader("학생 데이터")
            st.text_area("저장된 학생 데이터", student_data, height=300)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

# 푸터
st.markdown("""
---
**학생 안내**: 이 데모는 학습 목적의 간단한 예시입니다. 실제 딥페이크 탐지에는 **LSTM**, **Transformer** 같은 고급 모델과 대규모 데이터셋이 필요합니다.  
이 앱을 통해 **AI 윤리**, **허위 정보 대응**, **미디어 리터러시** 등에 대해 더 깊게 배우고 생각해 보세요.
""")
