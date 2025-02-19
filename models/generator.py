from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from retriever import *
import requests
import torch
import re
import math
import nltk
import json
import numpy as np
import os

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        # event:result에 해당하는 data만 저장할 리스트
        store_data = []

        with requests.post(
            self._host + '/testapp/v1/chat-completions/HCX-DASH-001',
            headers=headers,
            json=completion_request,
            stream=True
        ) as r:
            is_result_event = False  # 직전 라인이 event:result인지 판별하기 위한 플래그

            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()

                    # event:result 라인이 나오면 플래그 설정
                    if decoded_line.startswith("event:result"):
                        is_result_event = True

                    # event:result 바로 다음에 오는 data 라인일 경우 데이터를 추출하여 저장
                    elif decoded_line.startswith("data:") and is_result_event:
                        data_content = decoded_line[len("data:"):].strip()
                        store_data.append(data_content)
                        is_result_event = False

        # 여기서는 event:result로 표시된 data만 store_data에 담겨있습니다.
        # 필요에 따라 반환 혹은 다른 처리 가능
        return store_data

def load_prompt(file_name):
    prompt_file_path = os.path.join('prompts', file_name)
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    return prompt_text


def ask_llm_corpus(query: str, prompt_text: str, completion_executor) -> str:
    """
    query: 사용자 질문(예: "동학 농민 운동")
    prompt_text: RAG로 검색한 문서 정보를 합쳐서 만든 텍스트(build_prompt의 결과 등)
    completion_executor: CompletionExecutor 인스턴스(Clova API 호출용)

    이 함수를 통해 'assistant_content'만 반환
    """
    
    # 프롬프트 불러오기
    completion_prompt = load_prompt("completion_prompt.txt")
    completion_prompt = completion_prompt.format(prompt_text=prompt_text)

    preset_text = json.loads(completion_prompt)

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1600,
        'temperature': 0.7,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    # 1) API 호출
    response_string_list = completion_executor.execute(request_data)

    # 2) 응답이 비어있으면 에러 메시지 반환
    if not response_string_list:
        return "죄송합니다, 응답을 받지 못했습니다."

    # 3) 첫 번째 응답을 JSON 파싱
    json_str = response_string_list[0]
    parsed_response = json.loads(json_str)

    # 4) assistant_content 추출
    assistant_content = parsed_response["message"]["content"]

    return assistant_content



def convert_part_to_kkokkomu(part, tokenizer, persona_model):
    """
    part를 여러 문장으로 분할한 뒤,
    각 문장을 '말투만' 변환하고 합치는 방식.
    """

    # 1) 문장 분할 (예: 마침표/물음표/느낌표 뒤 공백 기준)
    #    정교한 처리를 위해 re.split로 문장 부호를 포함하도록 할 수도 있음.
    #    여기서는 간단히 '문장 끝 부호 + 공백' 정도로 시도.
    #    만약 마침표 뒤에 바로 줄바꿈이 있을 수 있으니, \s+ 활용
    sentences = re.split(r'(?<=[.!?])\s+', part.strip())

    transformed_sentences = []
    few_shot_examples = [
    {   "input": "1999년, 10월 대전에서 수연 씨는 저녁을 준비 중이었다.",
        "output": "때는 1999년 10월 대전의 어느 가정 집이야. 수연 씨는 맛있는 저녁을 준비 중이었어." },

    {   "input": "1453년(단종 1년), 김종서와 황보인 등이 수양대군(세조)을 제거하려 한다는 혐의로 사사 당하였습니다.",
        "output": "때는 1453년. 단종 1년이였어. 수양대군, 즉 세조를 제거하려 한다는 김종서와 황보인이 제거당했어." },
    {
        "input": "사건 당시 차 키는 꽂혀 있지 않았다. 또한, 피해자인 경태의 혈흔이 조수석 손잡이와 뒷좌석에서 발견되었다",
        "output": "운전을 하려면 반드시 필요한 거지? 그런데 차 키가 꽂혀 있지 않았던 거야. 게다가 조수석 손잡이와 뒷좌석에서 경태 씨의 혈흔이 발견돼."
    },
    {
        "input": "초기 수사에서 형사들은 이 사건이 낚시터에서 발생한 강도 살인일 가능성이 있다고 판단했다. 하지만, 차량 내부에서는 범인의 흔적이 전혀 발견되지 않았다.",
        "output": "형사들은 처음에는 낚시터에서 벌어진 강도 살인일 거라고 생각했어. 하지만 차 안에선 범인의 흔적은 전혀 발견되지 않았어."
    },
    {
        "input": "당시에는 CCTV나 블랙박스가 흔하지 않았으며, 목격자도 존재하지 않아 범인을 특정할 수 있는 증거가 부족했다.",
        "output": "자, CCTV, 블랙박스? 흔치 않던 시대야. 목격자? 없어. 범인을 특정할 증거가 없는 거야."
    },
    {
        "input": "부검 결과 피해자의 사인은 불명으로 판정되었으며, 정확한 사망 원인을 알 수 없었다. 이에 따라 형사들은 단순 강도 사건이 아닌 계획된 살인 가능성을 염두에 두고 수사를 진행하기로 했다.",
        "output": "피해자의 시신에 뭔가 남아 있을 수도 있잖아. 사인 불명. 원인을 알 수 없다는 거야. 형사들은 단순 강도 살인이 아닌 계획 살인을 염두해 두고 수사를 하기로 했어."
    },
    {
        "input": "피해자는 배우자와 만난 지 석 달 만에 혼인신고를 했으며, 이후 곧바로 동거를 시작했다.",
        "output": "만난 지 석 달 만에 혼인신고만 하고 동거를 시작했다는 거야."
    },
    {
        "input": "수사 과정에서 피해자의 생명보험을 조회한 결과, 총 여섯 개의 보험이 가입되어 있는 것이 확인되었다.",
        "output": "피해자의 보험을 조회해 보니 총 여섯 개가 가입되어 있었어."
    },
    {
        "input": "두 사람은 10월 2일에 혼인신고를 했으며, 보름 후부터 신혼 생활을 시작했다.",
        "output": "두 사람이 혼인신고를 한 날짜는 10월 2일이야. 자, 혼인 신고를 한 지 보름 후부터 두 사람은 신혼 생활을 시작했어."
    },
    {
        "input": "이후 아내인 수연 씨는 2주 동안, 즉 10월 28일부터 11월 3일까지 남편 명의로 총 여섯 개의 생명보험을 추가로 가입했다.",
        "output": "그러면서 아내 수연 씨는 당장 그 주부터 2주에 걸쳐 28일, 11월 3일까지, 이렇게 남편 앞으로 여섯 개의 생명보험을 가입해."
    },
    {
        "input": "그리고 그로부터 며칠 후, 남편이 실종되었으며, 결국 사망한 채 발견되었다.",
        "output": "그리고. 그 며칠 후에 남편이 사라졌고. 결국 남편은 사망한 채 발견이 된 거야."
    },
    {
        "input": "이 모든 사건은 두 달도 채 되지 않는 기간 동안 발생했다.",
        "output": "자, 요렇게 두 달이 채 안 되는 기간에 벌어진 일이야."
    }
    ]
    for sentence in sentences:
        # 불필요한 빈 문장 무시
        if not sentence.strip():
            continue

        # 2) 각 문장마다 프롬프트 생성
        base_prompt = load_prompt("kkkomu_prompt.txt")

        # Few-shot 예제 추가
        for example in few_shot_examples:
            prompt = base_prompt + f"원본: {example['input']}\n"
            prompt = base_prompt +  f"말투 변환: {example['output']}\n\n"

        # 변환할 '한 문장' 추가
        prompt += f"""이제 아래 한 문장 원본을 말투 변환된 스타일로 바꿔줘:
        원본: {sentence}\n말투 변환:"""

        # 3) 모델 실행
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = persona_model.generate(
            input_ids,
            #max_length=20000,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=400
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 4) '말투 변환:' 이후 결과만 추출
        if "말투 변환:" in output_text:
            converted_line = output_text.split("말투 변환:")[-1].strip()
        else:
            converted_line = output_text.strip()

        # 5) 영어 알파벳 제거 (정규식)
        converted_line = re.sub(r'[A-Za-z]', '', converted_line)

        transformed_sentences.append(converted_line)

    # 최종적으로 문장들을 공백(또는 줄바꿈)으로 이어붙이기
    final_result = " ".join(transformed_sentences)

    return final_result


def split_text_into_parts(text, max_parts=5):
    """
    1️⃣ NLTK로 문장을 분할
    2️⃣ 문장 임베딩을 생성하여 의미적으로 가까운 문장들을 묶음
    3️⃣ Clustering을 활용하여 max_parts 개수만큼 그룹화 (순서 유지)
    4️⃣ 최종적으로 max_parts 개의 그룹을 순서대로 정리하여 반환
    """
    nltk.download("punkt")
    nltk.download('punkt_tab')
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)

    if num_sentences <= max_parts:
        return sentences  # 전체 문장 그대로 반환

    # 문장 임베딩
    # SentenceTransformer 모델 로드 (의미 유사도 계산)
    embedding_model_cluster = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence_embeddings = embedding_model_cluster.encode(sentences)

    # Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=max_parts, linkage="ward")
    clusters = clustering.fit_predict(sentence_embeddings)

    # 클러스터별로 문장 묶기
    clustered_parts = {i: [] for i in range(max_parts)}
    for idx, cluster_id in enumerate(clusters):
        clustered_parts[cluster_id].append(sentences[idx])

    # 원래 순서를 유지하면서 문장 정렬
    ordered_parts = []
    seen_sentences = set()
    for sentence in sentences:
        for cluster_id, group in clustered_parts.items():
            if sentence in group and sentence not in seen_sentences:
                ordered_parts.append(sentence)
                seen_sentences.add(sentence)

    # max_parts 개수만큼 균등하게 분할
    sentences_per_part = math.ceil(len(ordered_parts) / max_parts)
    final_parts = []
    for i in range(0, len(ordered_parts), sentences_per_part):
        joined_part = " ".join(ordered_parts[i : i + sentences_per_part])
        final_parts.append(joined_part)


    print(final_parts[:max_parts])
    return final_parts[:max_parts]


def generate_story_outline(user_input, max_parts=5, tokenizer=None, persona_model=None, faiss_store=None, bm25=None, docs=None, embeddings=None):
    """
    1. 사용자 입력에 따라 텍스트 준비
    2. 문장 유사도 기반 분할 (split_text_into_parts)
    3. 각 파트를 꼬꼬무 스타일로 변환 (convert_part_to_kkokkomu)
    4. 최종 리스트 반환
    """
    load_dotenv()
    
    completion_executor = CompletionExecutor(
            host='https://clovastudio.stream.ntruss.com',
            api_key=os.getenv("Api_key"),
            request_id='6dc56592f44747789164dc55eb86227a'
    )
    final_docs = hybrid_search(
        query=user_input,
        faiss_store=faiss_store,
        bm25=bm25,
        docs=docs,
        embeddings=embeddings,
        alpha=0.35,
        top_k=4
        )
    prompt_text = build_prompt(final_docs, user_input)

    text = ask_llm_corpus(user_input, prompt_text, completion_executor)

#     text = """
# """
    # 1) 문장 분할
    parts = split_text_into_parts(text, max_parts)

    # 2) 각 파트 꼬꼬무 말투 변환
    converted_parts = []
    for p in parts:
      #print(p)
      converted = convert_part_to_kkokkomu(p,tokenizer, persona_model)
      converted_parts.append(converted)

    return converted_parts


def generate_story_segment(
    user_response: str = "",
    prev_assistant_response: str = "",
    tokenizer: AutoTokenizer = None, 
    persona_model: AutoModelForCausalLM = None
) -> str:
    
    base_prompt = load_prompt("kkkomu_prompt.txt")

    reaction_few_shot = [
    {
        "user": "헉, 진짜 실화야?",
        "assistant": "그렇지, 실화야."
    },
    {
        "user": "이거 너무 무섭다...",
        "assistant": "무섭지."
    },
    {
        "user": "와, 진짜 대단하네",
        "assistant": "그러게, 대단하지."
    },
    {
        "user": "설마 이게 다 계획이었어?",
        "assistant": "맞아, 소름이지."
    },
    {
        "user": "그래서 어떻게 됐어?",
        "assistant": "계속 지켜봐!"
    },
    {
        "user": "이게 말이 돼?",
        "assistant": "나도 믿기 힘들어."
    },
    {
        "user": "내가 생각한 게 맞나?",
        "assistant": "맞아."
    },
    {
        "user": "진짜 끔찍하네...",
        "assistant": "무섭지?"
    },
    {
        "user": "결국 어떻게 됐어?",
        "assistant": "이제 시작이야!"
    },
    {
        "user": "헐, 너무 충격이야",
        "assistant": "충격적이지!"
    }
    ]
    
    prompt = base_prompt.format(prev_assistant_response=prev_assistant_response, user_response=user_response)


    # Few-shot 예시 추가 (reaction_few_shot 등)
    for ex in reaction_few_shot:
        prompt += f"사용자: {ex['user']}\n해설자: {ex['assistant']}\n\n"

    # 실제 생성 요청
    prompt += "해설자:\n짧게 답변:"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = persona_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "짧게 답변:" 이후만 파싱
    answer = generated_text.split("짧게 답변:")[-1].strip()

    return answer