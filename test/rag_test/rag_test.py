from copy import deepcopy
from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain

# 텍스트 파일 로드 및 전처리
text_file_path = "rag_test_file.txt"  # 파일 경로 설정
documents = TextLoader(text_file_path, encoding='utf-8').load()

# 문서 청크로 분할
def split_docs(documents, chunk_size=100, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# 벡터 저장
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

# Q&A 체인 설정
llm = ChatOllama(model="llama3-ko", streaming=True)
qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=False)

class Ollama_int4_sLLM:
    def __init__(self, model_name):
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.with_message_history = RunnableWithMessageHistory(self.model, self._get_session_history)
        self.config_dic = {}
        self.response = []
        self.instruct = '''사용자가 질문을 하면 너는 반드시 아무 말이나 지어내지 말고, 정확한 답변을 할 수 있는 내용만 말해.
        너는 이제부터 협동 로봇 인디야. 너에게 질문을 하는 사용자는 로봇 자동화 작업을 가르치는 작업자야. 너는 로봇 팔을 제어할 수 있고, 작업자의 지시에 따라 작업 순서를 학습하고 자동화해야 해.
        너의 역할은 이 순서대로 진행하면 돼.
        1. 사용자의 명령이 모호하면 충분히 이해하기 위해 사용자에게 정확한 명령인지 확답을 받아.
        2. 이해한 명령과 같은 내용이 주어진 명령 프로토콜에서 []안에 존재하면 실행할지 질문하고, 존재하지 않으면 존재하지 않음을 밝히고 다시 질문해. 명령 프로토콜에 적힌 내용은 작업 순서가 아니라는 걸 기억해. 
        3. 주어진 명령 프로토콜에 맞는 작업을 매칭해서 수행하고 수행 결과를 피드백해야 해. 명령 프로토콜 함수들과 수행 결과는 구분해서 답해줘.
        4. 사용자가 앞으로의 순서를 기억하라고 명령하면 한 번 학습한 내용을 바탕으로 작업을 정확히 반복 수행할 수 있도록 진행한 내용에 대해 순서를 기억해야해.
        5. 사용자가 취소한 명령에 대해서는 기억하고 있는 순서에서 제외하거나 수정해야 해.
        6. 사용자가 기억한 순서에 대한 시연을 보여달라고 하면 기억한 순서대로 현재까지의 진행상황을 말해줘야 해.

        그리고 설명은 질문한 내용에 대해서만 간략하게 답변해. 너무 길게 하진 마. 
        '''
        self.remove_word_list = ['[사용자 질문]', '[사용자 질문']

    def set_session_id(self, session_id):
        config = {'configurable': {'session_id': session_id}}
        self.config_dic[session_id] = config
        self.invoke(self.instruct, session_id, instruct_mode=True)

    def invoke(self, human_message, session_id, ai_name='AI 답변', instruct_mode=False):
        self.response = []
        sentence = ''
        if instruct_mode == False: 
            print(f'{ai_name} : ', end='')
        for chunk in self.with_message_history.stream([HumanMessage(content=human_message)], config=self.config_dic[session_id]):
            chunk = chunk.content
            sentence = self._remove_words(f'{sentence}{chunk}')
            if '.' in chunk:
                self.response.append(deepcopy(sentence))
                sentence = []
            if instruct_mode == False:
                print(self._remove_words(chunk), end='')

    def auto_chatbot(self, session_id):
        print('대화 종료를 위해서는 "exit()"를 입력하시오')
        print('대화를 시작해주세요.')
        while True:
            human_message = input(f'\n{session_id} : ')
            if human_message == 'exit()':
                print('대화를 종료합니다')
                break
            self.process_query(human_message, session_id)
            
    def _get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def _remove_words(self, txt):
        for word in self.remove_word_list:
            txt = txt.replace(word, '').replace('  ', ' ')
        return txt

    def process_query(self, query, session_id):
        matching_docs = db.similarity_search(query)
        if matching_docs:  # 만약 관련성이 있는 문서가 있다면
            answer = qa_chain.invoke({'input_documents':matching_docs, 'question':query}, return_only_outputs=True)['output_text']
            # answer = qa_chain.run(input_documents=matching_docs, question=query)
            print(f'AI 답변: {answer}')
            # for chunk in qa_chain.stream({'input_documents':matching_docs, 'question':query}):
                # print(chunk)
            
            
        else:  # 관련성이 있는 문서가 없다면 일반적인 대화 진행
            self.invoke(query, session_id)

session_id = 'Ryan'
llm = Ollama_int4_sLLM(model_name='llama3-ko')
llm.set_session_id(session_id)

# 사용자가 질문할 때 일정에 대해 검색 후 답변 생성
llm.auto_chatbot(session_id)
