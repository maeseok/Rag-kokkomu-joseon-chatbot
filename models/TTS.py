import os
import urllib.request
import time
import re
import IPython.display as ipd  # Colab에서 오디오 재생용
import os
from dotenv import load_dotenv

def split_sentences(text):
    """ 문장을 마침표, 느낌표, 물음표 기준으로 나누기 """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # 문장 단위 분리
    return [s for s in sentences if s]  # 빈 문장 제거


def text_to_speech_single(
    text, speaker="ndaeseong", volume="0.7", speed="1.5", pitch="0", format="mp3", base_delay=1.0
):
    """
    전체 문장을 한 번에 TTS로 변환하여 출력하는 함수
    """
    
    # 네이버 클라우드 플랫폼 API 인증 정보
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    
    print(f"▶ 변환 중: {text}")
    # 텍스트 URL 인코딩
    encText = urllib.parse.quote(text)
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", CLIENT_ID)
    request.add_header("X-NCP-APIGW-API-KEY", CLIENT_SECRET)

    # 사용자 설정된 옵션 적용
    data = (
        f"speaker={speaker}&volume={volume}&speed={speed}&pitch={pitch}&format={format}&text={encText}"
    )

    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()

    if rescode == 200:
        output_file = "combined_tts_output.mp3"
        with open(output_file, 'wb') as f:
            f.write(response.read())
        return output_file  # 단일 MP3 파일 경로 반환
    else:
        print(f"❌ API 호출 오류 발생: {rescode}")

    return None