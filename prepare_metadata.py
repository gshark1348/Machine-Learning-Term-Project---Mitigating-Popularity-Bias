"""MovieLens 20M ZIP 에서 데모에 필요한 메타데이터 파일 두 개를 만든다.

사용법
    python prepare_metadata.py <ml-20m.zip 경로>

생성 파일 (이 스크립트가 있는 폴더에 저장)
    genome-tags.csv : 태그 사전 (tagId, tag). discovery.py 가 영화의 대표 태그 이름을 붙일 때 사용.
    poster_ids.json : {"movieId": "tt<imdbId>"} 형태. app.py 가 포스터 URL 을 만들 때 사용.

포스터 이미지는 외부 서비스 images.metahub.space 에서 IMDb id 로 불러온다(app.py 참고).
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

# 이 파일이 있는 폴더 (실행 위치와 상관없이 같은 곳에 결과 저장)
root = Path(__file__).resolve().parent

# 명령줄 첫 번째 인자(sys.argv[1])가 ZIP 경로. ZipFile(경로)는 기본 읽기 모드로 연다.
with zipfile.ZipFile(sys.argv[1]) as archive:
    # archive.read(이름): ZIP 안의 파일 하나를 압축 해제해 bytes 로 한 번에 반환.
    # 태그 사전은 내용 변경 없이 그대로 복사한다.
    (root / 'genome-tags.csv').write_bytes(archive.read('ml-20m/genome-tags.csv'))

    # archive.open(이름): 파일을 통째로 읽지 않고 스트림(바이트 파일 객체)으로 연다. links.csv 는 큰 편.
    with archive.open('ml-20m/links.csv') as source:
        # io.TextIOWrapper(바이트 스트림): bytes 를 문자열로 디코딩해 주는 텍스트 파일 객체로 감싼다
        #   (csv 모듈은 문자열 줄을 요구하기 때문).
        # csv.DictReader(텍스트): 첫 줄을 열 이름으로 보고 각 행을 {열이름: 값} dict 로 읽는다.
        rows = csv.DictReader(io.TextIOWrapper(source))
        # imdbId 가 비어 있지 않은 영화만. IMDb id 는 앞에 'tt' 를 붙여 'tt0114709' 같은 형식으로 만든다.
        ids = {r['movieId']: 'tt' + r['imdbId'] for r in rows if r['imdbId']}

    # json.dumps(객체): 파이썬 dict 를 JSON 문자열로 바꾼다 (write_text 로 파일에 저장).
    (root / 'poster_ids.json').write_text(json.dumps(ids))
