from models.pipeline import * 
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

def main():
    story_progress = 0
    story_segments = []
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 해설자 모델 및 토크나이저 로드
    explainer_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    persona_model = AutoModelForCausalLM.from_pretrained(explainer_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(explainer_model_name)
    
    # 기존 코드: docs, faiss_store, bm25, embeddings 준비
    docs_directory = ""

    # 2) 단일 파일로 합쳐 청킹
    docs = load_docs_as_single_txt(
        directory=docs_directory,
        chunk_size=700,
        overlap=50
    )
    # FAISS 인덱스
    faiss_store = build_faiss_index(docs)
    # BM25 인덱스
    bm25 = build_bm25_index(docs)

    # 같은 임베딩 객체를 하이브리드로도 사용
    embedding_model = "intfloat/multilingual-e5-large"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    css = """
    .gradio-container {
        background: linear-gradient(to top, #fff, #ededed);
        font-family: 'Noto Sans KR', sans-serif;
    }

    #chatbot-box {
        background-color: #bacee0 !important;
        padding: 10px;
    }
    #chatbot-box .message.user {
        background-color: #fae105;
        color: #000;
        border-radius: 10px;
        text-align: right;
        margin-left: auto;
        margin-top: 5px;
        margin-bottom: 5px;
        padding: 8px 12px;
    }
    #chatbot-box .message.bot {
        background-color: #ffffff;
        color: #000;
        border-radius: 10px;
        text-align: left;
        max-width: 60%;
        margin-right: auto;
        margin-top: 5px;
        margin-bottom: 5px;
        padding: 8px 12px;
    }

    #bottom-input-row {
        width: 100%;
        display: flex;
        gap: 8px;
        margin-top: 10px;
    }

    #user-input {
        flex: 1;
        border-radius: 4px;
        border: 1px solid #ccc;
        padding: 8px;
        height: 60px;
    }

    #send-button {
        width: 20px;
        border: none;
        border-radius: 4px;
        background-color: #ffcd00;
        font-weight: bold;
        cursor: pointer;
        height: 60px;
    }
    #send-button:hover {
        background-color: #ffdd33;
    }

    .chat-header img {
            width: 40px;
            height: 40px;
            margin-right: 10px;
    }

    .chat-header {
        text-align: center;
        background-color: #FEE500;
        padding: 10px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid #3B1E1E;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }

    .progress-text, .meta-text {
        display: none !important;
    }

    .audio-container {
        height: 343px;  /* 필요에 따라 조정 */
        display: flex;
        align-items: center; /* 수직 정렬 */
    }

    .left-panel {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%; /* 부모 컨테이너의 높이를 가득 채우도록 설정 */
    }


    """
    # Gradio 인터페이스 구성
    with gr.Blocks(css=css) as demo:
        with gr.Row():
            gr.HTML("""
            <div class='chat-header'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/e/e3/KakaoTalk_logo.svg'>
                <h1 style='margin: 0;'>꼬리에 꼬리를 무는 조선왕조실톡</h1>
            </div>
            """)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 안내사항")
                gr.Markdown("- 첫 입력: '주제어'를 입력하면 해당 주제로 이야기가 시작됩니다.")
                gr.Markdown("- 이후 메시지: 이야기를 발전시키거나 질문을 할 수 있습니다.")
                gr.Markdown("- 일정 횟수 입력 후 이야기가 자동 종료됩니다.")
                audio_player = gr.Audio(interactive=True)

            with gr.Column(scale=7):
                chatbot_ui = gr.Chatbot(elem_id="chatbot-box", label="이야기 진행 상황")
                with gr.Row(elem_id="bottom-input-row"):
                    txt_input = gr.Textbox(
                        show_label=False,
                        placeholder="메시지를 입력하세요...",
                        elem_id="user-input"
                    )
                    btn_send = gr.Button("전송", elem_id="send-button")

        def process_chat(user_input, history):
            chat_updates = list(chatbot(user_input, history, tokenizer, persona_model, story_progress, story_segments, faiss_store, bm25, docs, embeddings))
            if chat_updates:
                new_history, _ = chat_updates[-1]  # 마지막 업데이트된 채팅 기록 사용
            else:
                new_history = history  # 빈 응답 방지
            return new_history, "combined_tts_output.mp3", ""

        txt_input.submit(process_chat, [txt_input, chatbot_ui], [chatbot_ui, audio_player, txt_input], queue=False)
        btn_send.click(process_chat, [txt_input, chatbot_ui], [chatbot_ui, audio_player, txt_input], queue=False)

    demo.launch(debug=True, share=True)