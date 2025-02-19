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


def ask_exaone_corpus(query: str, prompt_text: str, completion_executor) -> str:
    """
    query: 사용자 질문(예: "동학 농민 운동")
    prompt_text: RAG로 검색한 문서 정보를 합쳐서 만든 텍스트(build_prompt의 결과 등)
    completion_executor: CompletionExecutor 인스턴스(Clova API 호출용)

    이 함수를 통해 'assistant_content'만 반환
    """

    preset_text = [
        {
            "role": "system",
            "content": f"""당신은 조선왕조실록 전문가입니다.
            <작업 지시>
            아래에 제공된 문서 내용(토픽: '{query}')을 바탕으로, 정확하고 구체적인 역사적 배경을 구성해 주세요.
            **하나의 단락**으로 작성하되, 총 **800자에서 1000자** 사이가 되어야 합니다.
            글을 시작할 때, **연도나 시점을 언급**하여 시간정보를 제공하세요.
            **문서를 직접 해설**하듯이, 섹션 구분 없이 자연스럽게 이어지는 문장들로 구성해 주세요.
            **문서에 언급된 인물(예: 세종, 태종, 장영실 등), 사건(훈민정음 창제, 왕세자 대리청정 등), 연도(1443년, 1436년 등), 제도, 구체적 일화** 등을 반드시 포함하십시오.
            문서에 나오는 **인용구**(예: "내가 두 눈이 흐릿하고 깔깔하며...")나 **출처**(예: [세종실록] 92권, 23년(1441)...)를 적절히 활용하여, 추상적 요약이 아닌 **구체적 사례**를 들어주세요.
            **시간 흐름 순서**에 따라 독자가 이해하기 쉽게 기술하십시오.
            **문서 밖의 추측이나 다른 출처 내용은 추가하지 말고**, 주어진 문서 정보만 기반으로 정확히 작성해주세요.
            만약 문서에 실린 조선왕조실록 및 한국사 내용과 '{query}'의 실제 내용이 상충되는 정보가 있다면, 해당 내용은 **무시**하십시오.


            """
        },
        {
            "role": "user",
            "content": f'''문서: {prompt_text}

            반드시 답변은 800글자에서 1000글자 사이의 줄글형식으로 해주세요.

            예시 : 세종 25년(1443년), 세종은 독서광으로 한시도 손에서 책을 놓지 않았다. 그러던 어느 날, 세종은 심한 몸살로 열이 펄펄 끓으면서도 책을 읽었고, 이 모습을 본 아버지 태종이 그의 책을 모두 거둬오라는 명을 내린다. 하지만 세종은 병풍 틈에 숨겨놓은 책 한 권을 발견하고선 그 책만 수없이 반복해 읽었다.\r\n
            세종은 일 중독자로, 지나치게 높은 학구열과 격무로 인해 40대에 건강이 안좋아진다. 특히 시력이 급격히 떨어져, '내가 두 눈이 흐릿하고 깔깔하며 아파, 봄부터는 음침하고 어두운 곳은 지팡이가 아니고는 걷기가 어려웠다.' 라는 기록이 남아있을 정도였다.\r\n
            하지만 세종은 왕세자에게 결재권을 넘겨주고 자신은 현업에서 물러나겠다는 뜻을 내비쳤음에도 불구하고, 여전히 격무를 수행했다. 오히려 몸이 아플 때, 더욱 놀라운 업적을 쌓으니, 그게 바로 '훈민정음' 창제였다.\r\n
            1443년, 세종은 백성을 지극히 사랑하는 마음으로 한글을 창조했다. 당시 지배층들은 글을 읽고 쓰는 것을 자신들의 특권으로 여겼지만, 세종은 이를 원치 않았다. 일반 백성들은 글을 모르니 학문을 익힐 수 없었고, 그로 인해 어리석은 죄를 저지르게 되는 것이 싫었던 것이다.  \r\n
            한글은 누구나 쉽게 배우고 쓸 수 있으며, 자기의 의사를 마음대로 표현할 수 있을 뿐만 아니라, 글자를 만드는 원리가 매우 과학적인 뛰어난 문자이다. 실록에서는 한글의 창제원리에 대한 언급이 없지만, 1940년에 발견된 훈민정음 해례본 에 한글 창제원리가 자세히 적혀 있다. \r\n
            세종은 한글을 보급하기 위해 용비어천가, 석보상절, 월인천강지곡 등의 책을 한글로 출판하였다.  이처럼 지칠 줄 모르는 학구열을 지닌 세종은 몸이 아파도 멈추지 않고, 백성을 사랑하는 마음으로 한글을 창제했다. 1443년에 이루어진 이 업적은 누구나 쉽게 글을 배울 수 있도록 하여, 당시 백성들의 삶을 크게 바꾸었다. 정인지 등의 학자들이 한글 책과 해설서를 보급하며 '슬기로운 이는 하루아침에도 깨치고, 어리석은 이도 열흘이면 배운다'는 말을 현실로 만들었다. 그 결과 한글은 오늘날까지도 독창성과 과학성을 인정받는 소중한 문화유산으로 남아 있다.\r\n

            답변:

            '''
        }
    ]

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
        prompt = f"""너는 대한민국의 원본 문장의 말투를 바꾸는 최고의 해설자야.
        너의 역할은 원본 문장의 주어진 내용을 절대 변경하지 않고, 말투만 바꾸는 것이야.
        절대 내용을 바꾸면 안 돼! 정보를 그대로 유지하면서 말투만 변환해야 해.

        말투 변환 방식:
        - 무조건 반말을 사용해야 해.
        - 정보를 나열하는 방식이 아니라, 이야기를 하듯이 자연스럽게 말해야 해.
        - 실제 해설자가 말하는 것처럼 이어서 서술해야 해.
        - 반전, 감탄사, 청중의 호응을 이끌어내는 표현을 추가해야 해.
        - 너는 영어를 배운 적이 없어. 절대 영어를 사용하지 마.

        예시는 다음과 같아:

        """

        # Few-shot 예제 추가
        for example in few_shot_examples:
            prompt += f"원본: {example['input']}\n"
            prompt += f"말투 변환: {example['output']}\n\n"

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

    text = ask_exaone_corpus(user_input, prompt_text, completion_executor)

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
    
    prompt = f"""
    # 이전 해설자 응답 (맥락):
    {prev_assistant_response}

    # 사용자 반응:
    {user_response}

    너는 이야기를 계속 들려주는 해설자야. 사용자의 반응이 질문이면 간단히 확인하거나 맞장구쳐 주고,
    감정 표현이면 짧게 공감해 줘.

    조건:
    - 반말
    - 문장 끝에 감탄사나 여운
    - 최대 10자 이내, 한 문장
    - 절대 영어 쓰지 마
    - 대화 흐름을 자연스럽게 이어가

    아래는 참고 예시:
    """

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