import getpass
import os

##정답에 가까운 코드는 day3 6번 챕터에 올라와있음


os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# from langchain import hub

# prompt = hub.pull("rlm/rag-prompt")

# example_messages = prompt.invoke(
#     {"context": "filler context", "question": "filler question"}
# ).to_messages()

# example_messages

# print("!!!! 0")
# print(example_messages)

###### 웹 로더

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

########## 1 => Docs Retrieval
########## 1단계 코드: 웹 문서 로드 및 청크화
# 이 단계에서는 웹에서 문서를 로드하고 이를 처리 가능한 형태로 변환합니다. 주어진 URL에서 데이터를 불러오는 방법을 구현합니다.

# 1. 웹 문서 로드
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# WebBaseLoader를 사용하여 웹 페이지에서 문서 로드
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")  # 필요한 텍스트만 추출
        )
    ),
)

# 문서 불러오기
docs = loader.load()

# 2. 문서 청크화 (문서를 적당한 크기로 나눔)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 어떻게 동작하는지 설명:
# **RecursiveCharacterTextSplitter**는 문서를 일정한 크기(chunk_size)로 나누는 클래스입니다.
# chunk_size=1000: 문서를 1000자 단위로 나누겠다는 의미입니다.
# chunk_overlap=200: 청크 사이에 200자의 중복을 두어 문서의 문맥을 유지하게 설정합니다.
# split_documents(docs):
# docs 변수에 저장된 불러온 문서들을 입력으로 받아, 이 문서들을 청크 단위로 나눕니다. 이렇게 나눠진 청크들은 splits 변수에 저장됩니다.
# 예를 들어, 한 문서가 3000자라고 하면, 첫 번째 청크는 11000자, 두 번째 청크는 8011800자, 세 번째 청크는 1601~3000자와 같이 나누어집니다.
# RecursiveCharacterTextSplitter를 사용하는 이유:
# 긴 문서 처리: 많은 경우, LLM(Large Language Model)은 한 번에 처리할 수 있는 토큰(문자) 수에 제한이 있습니다. 긴 문서는 한 번에 처리하기 어렵기 때문에, 적당한 크기로 나누는 것이 필요합니다.

# 문맥 유지: 청크를 나누는 동안 일정 부분의 중복을 포함시켜, 청크 사이의 연관성을 유지할 수 있습니다. 이 방식은 문서의 자연스러운 흐름을 유지하며 검색과 요약 등에 도움을 줍니다.

# 실제 적용된 사례:
# 만약 주어진 문서가 다음과 같다면:

# "This is a long document about natural language processing. It contains information about task decomposition, large language models, embeddings, and more."
# 이를 청크화하면 다음과 같이 나누어집니다:

# 첫 번째 청크: "This is a long document about natural language processing. It contains..."
# 두 번째 청크: "It contains information about task decomposition, large language models..."
# 이렇게 청크된 문서는 이후에 임베딩, 검색, 요약 등을 할 때 효율적으로 처리됩니다.

# 따라서, 1단계에서는 문서를 잘라서 검색할 준비를 하고, 다음 단계에서는 이 청크들에 대해 임베딩을 생성한 후 벡터 스토어에 저장하여 검색 가능하도록 합니다.

##########################################
# 로드된 문서와 청크 결과 확인
#print(f"Loaded {len(docs)} documents.")
#print(f"Chunked into {len(splits)} parts.")
## 결과 
# Loaded 3 documents.
# Chunked into 183 parts.
##########################################

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 임베딩 모델 설정 (OpenAI 임베딩 사용)
# "text-embedding-3-small"이라는 임베딩 모델을 사용한다면 아래 모델 이름을 대체하세요.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 모델 이름은 필요에 따라 조정

# 2. Chroma 벡터 스토어에 문서 청크 저장
vectorstore = Chroma.from_documents(
    documents=splits,  # 청크된 문서들
    embedding=embeddings  # 임베딩 모델
)

# 3. 벡터 스토어 저장 확인
print(f"Successfully stored {len(splits)} chunks in the vector store.")
### 결과
# Successfully stored 183 chunks in the vector store.