from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import numpy as np
import os
import glob
import re


def natural_sort_key(filename: str):
    """
    파일명 문자열에서 숫자를 찾아 정수 변환.
    예: "ocr_result_chunk_10.txt" -> ["ocr_result_chunk_", 10, ".txt"]
    -> 비교 시 10이 숫자로 처리되어 '2' < '10' 순서를 제대로 반영.
    """
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r'(\d+)', filename)
    ]

def merge_txt_files(directory: str) -> str:
    """
    주어진 디렉터리 안의 모든 .txt 파일을
    '자연 정렬(natural sort)'된 순서대로 읽어 하나의 문자열로 합쳐 반환.
    파일 사이에는 개행문자('\n')로 구분.
    """
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    # 자연 정렬
    txt_files_sorted = sorted(txt_files, key=natural_sort_key)

    merged_text = []
    for file_path in txt_files_sorted:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        merged_text.append(text)

    # 파일 사이마다 줄바꿈을 추가하여 합침
    combined_text = "\n".join(merged_text)
    return combined_text


def split_text_into_chunks(text: str, chunk_size=1024, overlap=200) -> list:
    """
    text를 chunk_size 씩 잘라서 리스트로 반환.
    각 청크 사이에 overlap 길이만큼 문자가 겹치도록 함.
    """
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += step  # 겹치기 고려하여 다음 청크 시작점 이동

    return chunks


def load_docs_as_single_txt(
    directory: str,
    chunk_size=1024,
    overlap=200
) -> list:
    """
    1) directory 내 모든 txt 파일을 사전순으로 하나의 문자열로 합침
    2) 합쳐진 단일 문자열을 chunk_size, overlap 기반으로 청킹
    3) [{'doc_id':0, 'file_name':'merged_all.txt', 'chunk_index':..., 'chunk_text':...}, ...] 형태로 반환
    """
    # 1) 모든 텍스트 합침
    combined_text = merge_txt_files(directory)

    # 2) 청킹
    chunks = split_text_into_chunks(combined_text, chunk_size, overlap)

    # 3) docs 생성
    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append({
            "doc_id": 0,  # 전체를 하나의 doc으로 취급
            "file_name": "merged_all.txt",
            "chunk_index": i,
            "chunk_text": chunk_text
        })
    return docs


def build_faiss_index(docs):
    """
    주어진 문서(docs)를 LangChain Document로 변환 후
    Faiss 인덱스를 생성 & 반환
    """
    # 1) 임베딩 모델 로드 (E5 모델 예시)
    embedding_model = "intfloat/multilingual-e5-large"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # 2) langchain Document로 변환
    #    => chunk_text를 page_content로, 나머지를 metadata로
    langchain_docs = []
    for d in docs:
        langchain_docs.append(Document(
            page_content=d["chunk_text"],
            metadata={
                "doc_id": d["doc_id"],
                "file_name": d["file_name"],
                "chunk_index": d["chunk_index"]
            }
        ))

    # 3) FAISS VectorStore 생성
    faiss_store = FAISS.from_documents(langchain_docs, embeddings)

    # (옵션) 로컬에 인덱스 저장
    # faiss_store.save_local("faiss_index")

    return faiss_store


def custom_tokenize_kiwi(text: str) -> list:
    """
    Kiwi 형태소 분석기로 한글 문서를 토큰화하는 예시 함수.
    - 필요하면 품사(Part-of-speech)별 필터링 가능
    - 여기서는 간단히 모든 토큰 .form 만 추출
    """
    # 1) 필요한 전처리 (소문자 변환, 불필요 문장부호 제거 등) 수행 가능
    #    예: text = text.lower() / re.sub(...) 등
    # Kiwi 형태소 분석기 초기화
    kiwi = Kiwi()
    # 2) Kiwi 토큰화
    tokens = []
    # kiwi.tokenize()는 Token(namedtuple)을 반환
    # [Token(form='세종대왕', tag='NNP', start=0, len=4), Token(form='께서', ...), ...]
    kiwi_tokens = kiwi.tokenize(text)

    # 3) 품사 필터링 없이 모든 토큰 form을 수집
    for token in kiwi_tokens:
        tokens.append(token.form)

    return tokens

def build_bm25_index(docs):
    tokenized_corpus = []
    for d in docs:
        # Kiwi 형태소 분석
        tokens = custom_tokenize_kiwi(d["chunk_text"])
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def hybrid_search(
    query,
    faiss_store,  # FAISS 인덱스를 다루는 객체 (langchain 등 사용)
    bm25,
    docs,  # [{'doc_id':..., 'chunk_index':..., 'chunk_text':...}, ...]
    embeddings,  # HuggingFaceEmbeddings (langchain)
    alpha,
    top_k
):
    """
    query에 대해:
      1) Faiss에서 top-N (~= top_k * 5 정도) 문서를 찾고, dense 점수(코사인 유사도) 계산
      2) BM25 점수(스파스)를 docs 전체에 대해 계산
      3) 두 점수를 각각 min-max 정규화 후, alpha 가중합으로 결합 -> 상위 top_k 선정

    - alpha: dense 점수와 BM25 점수를 어떻게 섞을지 결정 (0.0 ~ 1.0)
    - top_k: 최종적으로 반환할 문서 개수
    """

    # 1) query 임베딩 구하기
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # 2) Faiss 쪽 검색
    #    top_k * 5 or 50개 문서를 가져오고, similarity_search_with_score()를 통해
    #    (문서, dense_score) 리스트를 반환받음
    N = max(top_k * 40, 200)
    dense_results = faiss_store.similarity_search_with_score(query, k=N)
    # dense_results: List[(Document, float(similarity))]

    # 3) BM25 점수 계산
    #    docs 전체에 대해 query의 BM25 점수를 구함
    query_tokens = custom_tokenize_kiwi(query)
    bm25_scores = bm25.get_scores(query_tokens)  # length = len(docs)

    # 4) dense_results를 doc_id, chunk_index 기반으로 매핑
    #    => { (doc_id, chunk_index): dense_score }
    doc2dense = {}
    for (doc_obj, score) in dense_results:
        md = doc_obj.metadata
        d_id = md["doc_id"]
        c_idx = md["chunk_index"]
        doc2dense[(d_id, c_idx)] = score

    # 5) 모든 docs에 대해 dense score, bm25 score를 구한 뒤 lists로 저장
    dense_list = []
    sparse_list = []

    for i, d in enumerate(docs):
        d_id = d["doc_id"]
        c_idx = d["chunk_index"]

        # dense score
        dense_s = doc2dense.get((d_id, c_idx), 0.0)
        dense_list.append(dense_s)

        # bm25 score
        sparse_s = bm25_scores[i]
        sparse_list.append(sparse_s)

    # 6) min-max 정규화
    #    dense_list, sparse_list 각각 min–max 스케일링
    dense_array = np.array(dense_list, dtype=float)
    sparse_array = np.array(sparse_list, dtype=float)

    def minmax_scale(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        # 만약 모든 값이 동일하다면 (arr_max - arr_min == 0) -> 0으로 처리
        # 혹은 분모에 1e-9 더하기
        if arr_max - arr_min < 1e-9:
            return np.zeros_like(arr, dtype=float)
        return (arr - arr_min) / (arr_max - arr_min)

    dense_norm = minmax_scale(dense_array)
    sparse_norm = minmax_scale(sparse_array)

    # 7) 최종 점수 = alpha * dense_norm + (1 - alpha) * sparse_norm
    final_scores = alpha * dense_norm + (1 - alpha) * sparse_norm

    # 8) 상위 top_k 문서 선정
    #    (점수가 높을수록 더 관련있음)
    results = [(final_scores[i], i) for i in range(len(docs))]
    results.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for (_, idx) in results[:top_k]]

    # 9) 반환
    out = []
    for ti in top_indices:
        out.append(docs[ti])
    return out

def build_prompt(docs, query):
    """
    docs: List of dict, each dict has 'chunk_text'.
    user_query: str, 예) "동학 농민 운동과 신분제 폐지의 배경"
    => 적절히 연결한 Prompt 문자열을 구성해 반환
    """
    context_text = ""
    for i, d in enumerate(docs):
        context_text += f"\n[문서 {i}]\n{d['chunk_text']}\n"

    return context_text