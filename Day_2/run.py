import getpass
import os

##정답에 가까운 코드는 day3 6번 챕터에 올라와있음
os.environ['USER_AGENT'] = 'MyApp/1.0.0'

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
    #collection_name="rag-chroma" # 이름을 붙여두는게 좋은경우 -> 여러개 vectorstore를 만드는경우 구분, 디폴트는 벡터 스토어가 계속 append되므로 그런경우 다른 종류의 vector가 차원이 다르거나 한데 계속 append됨
)

# 3. 벡터 스토어 저장 확인
print(f"Successfully stored {len(splits)} chunks in the vector store.")
### 결과
# Successfully stored 183 chunks in the vector store.

##### 코드: 사용자 쿼리와 관련된 청크 검색
# 1. Chroma 벡터 스토어에 대한 검색기능을 사용하기 위해 retriever 설정
retriever = vectorstore.as_retriever()

# 2. 사용자 쿼리 설정
#user_query = "agent memory"
#user_query = "Fig 11 document" # test sample

# 3. 관련된 청크를 벡터 스토어에서 검색
#related_docs = retriever.get_relevant_documents(user_query)

# 4. 검색된 결과 출력
# for i, doc in enumerate(related_docs):
#     print(f"Document {i+1}:")
#     print(doc.page_content)
#     print("\n")
# 결과: 
# Document 1:
# Memory stream: is ...
# Document 2:
# LLM Powered Autonom ...
# ...

##### User query와 retrieved chunk 에 대해 relevance 가 있는지를 평가하는 시스템 프롬프트 작성: 
# retrieval 퀄리티를 LLM 이 스스로 평가하도록 하고, 관련이 있으면 {‘relevance’: ‘yes’} 관련이 없으면 {‘relevance’: ‘no’} 라고 출력하도록 함. ( JsonOutputParser() 를 활용 )

from langchain_core.output_parsers import JsonOutputParser

# JsonOutputParser 설정
output_parser = JsonOutputParser()

# 시스템 프롬프트 설정
system_prompt = """
You are an expert in evaluating the relevance between a user's query and a document chunk. Your task is to assess whether the retrieved chunk is relevant to the query.

Given the user's query and a retrieved document chunk, decide whether the chunk contains information relevant to answering the query.

- If the retrieved chunk is relevant to the query, return the JSON: {"relevance": "yes"}.
- If the retrieved chunk is not relevant to the query, return the JSON: {"relevance": "no"}.

### Evaluation Criteria:
1. Check if the document chunk contains any information directly related to the topic or keywords in the user's query.
2. Consider the context in the chunk and evaluate if it addresses the user's query sufficiently.
3. Base your evaluation purely on the information provided in the chunk, without adding extra knowledge.

### Input format:
- Query: "<USER_QUERY>"
- Document Chunk: "<DOCUMENT_CHUNK>"

### Output format:
Return your result in a JSON format with a "relevance" key.

### Example:

- Query: "agent memory"
- Document Chunk: "This document talks about agents using memory to store and recall actions."

Correct output: {"relevance": "yes"}

- Query: "task scheduling"
- Document Chunk: "This section covers agent memory structures and retrieval."

Correct output: {"relevance": "no"}
"""

# 함수: 쿼리와 문서 청크 간 관련성 평가
def evaluate_relevance(user_query, document_chunk):
    # 프롬프트 생성
    input_text = f'Query: "{user_query}"\nDocument Chunk: "{document_chunk}"'
    
    # LLM을 사용한 평가 (ChatOpenAI 사용)
    result = llm.invoke(input_text, system_prompt=system_prompt)
    
    # JsonOutputParser로 결과 파싱
    parsed_result = output_parser.parse(result)
    
    return parsed_result

# Relevance checker 함수 정의
from langchain_core.messages import SystemMessage, HumanMessage

def relevance_checker(query, doc_content):
    # 시스템 메시지와 사용자 메시지 생성
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=f'Query: "{query}"\nDocument Chunk: "{doc_content}"')
    
    # LLM을 사용한 평가
    result = llm.invoke([system_message, human_message])
    
    # JsonOutputParser로 결과 파싱
    parsed_result = output_parser.parse(result.content)
    
    return parsed_result['relevance']

# 관련성 검사 및 결과 출력
# for i, doc in enumerate(related_docs):
#     relevance = relevance_checker(user_query, doc.page_content)
#     print(f"Document {i+1}:")
#     print(f"Relevance: {relevance}")  # 'yes' 또는 'no' 출력
#     print(doc.page_content)
#     print("\n")

def summarize_relevance(related_docs, user_query):
    relevance_results = []
    for doc in related_docs:
        relevance = relevance_checker(user_query, doc.page_content)
        relevance_results.append(relevance)
        print(f"Document Relevance: {relevance}")
        print(doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)
        print("\n")

    if all(result == "no" for result in relevance_results):
        print("Overall Relevance: No")
        return "No"
    elif any(result == "yes" for result in relevance_results):
        print("Overall Relevance: Yes")
        return "Yes"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

def generate_answer(query, relevant_docs):
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant. Use the following context to answer the user's question. If you're not sure, say you don't know."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
    ])
    response = llm.invoke(prompt.format_prompt().to_messages())
    return response.content

def hallucination_check(query, answer, relevant_docs):
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an AI designed to detect hallucinations in answers. Compare the given answer to the context and question. If the answer contains information not supported by the context, or if it's not relevant to the question, classify it as a hallucination."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}\n\nAnswer: {answer}\n\nIs this answer a hallucination? Respond with 'Yes' if it is, or 'No' if it isn't. Explain your reasoning briefly.")
    ])
    response = llm.invoke(prompt.format_prompt().to_messages())
    return response.content

def rag_with_hallucination_check(query, max_attempts=3):
    related_docs = retriever.invoke(query)
    relevance_results = []
    relevant_docs = []

    for doc in related_docs:
        relevance = relevance_checker(query, doc.page_content)
        relevance_results.append(relevance)
        if relevance == "yes":
            relevant_docs.append(doc)
        print(f"Document Relevance: {relevance}")
        print(doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)
        print("\n")

    if not relevant_docs:
        print("No relevant documents found. Cannot answer the query.")
        return None

    for attempt in range(max_attempts):
        answer = generate_answer(query, relevant_docs)
        hallucination_result = hallucination_check(query, answer, relevant_docs)
        
        if "No" in hallucination_result:
            print(f"Answer generated (Attempt {attempt + 1}):")
            print(answer)
            print("\nHallucination check passed. Returning the answer.")
            return answer
        else:
            print(f"Attempt {attempt + 1} failed hallucination check. Retrying...")
            print(f"Hallucination check result: {hallucination_result}")

    print(f"Failed to generate a non-hallucinatory answer after {max_attempts} attempts.")
    return None

# 메인 실행 부분
#user_query = "What is agent memory in AI?"  # 예시 쿼리, 실제 사용 시 사용자 입력으로 대체 가능
user_query = input("프롬프트 입력: ")
final_answer = rag_with_hallucination_check(user_query)

if final_answer:
    print("\nFinal Answer to User:")
    print(final_answer)
else:
    print("\nUnable to provide a reliable answer to the query.")