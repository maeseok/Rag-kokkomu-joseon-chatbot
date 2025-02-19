from retriever import * 
from generator import *
from TTS import * 
import gradio as gr

# 챗봇 함수
def chatbot(user_input, chat_history=[], tokenizer=None, persona_model=None, story_progress=None, story_segments=[], faiss_store=None, bm25=None, docs=None, embeddings=None):
    if story_progress == 0:
        story_segments = generate_story_outline(user_input, tokenizer, persona_model, faiss_store, bm25, docs, embeddings)
        output = story_segments[story_progress]
        text_to_speech_single(story_segments[story_progress])
        story_progress = 1  # 다음 파트로 이동
    else:
        if story_progress < len(story_segments):
            reaction = generate_story_segment(user_input,story_segments[story_progress-1], tokenizer, persona_model)
            next_part = story_segments[story_progress]
            output = reaction +" "+ next_part
            text_to_speech_single(output)
            story_progress += 1
        else:
            # 이야기가 모두 끝나면 종료
            output = "✅ 오늘의 이야기는 여기까지야! (스토리 종료)"


    # (2) 우선 사용자 메시지를 챗 히스토리에 추가하되, 봇 답변은 아직 빈 문자열("")
    chat_history.append((user_input, ""))

    # Textbox를 비우기 위해 두 번째 출력에 gr.update(value="") 전달
    # 즉시 중간 업데이트(사용자 메시지는 바로 보이되, 봇 답변은 비어 있는 상태)
    yield chat_history, gr.update(value="")

    # (3) 봇의 답변을 한 글자씩 출력하기
    partial_response = ""
    for char in output:
        partial_response += char
        # chat_history의 마지막(봇의 답변)을 계속 갱신
        chat_history[-1] = (user_input, partial_response)
        yield chat_history, gr.update(value="")
        # 글자 나오는 속도(너무 길면 느림) 원하는 대로 조절
        import time
        time.sleep(0.02)