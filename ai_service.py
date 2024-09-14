from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ==============================================
#               OpenAPI LLM
# ==============================================
# Initialize LLM at server initialization
llm = ChatOpenAI(model="gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "heavily use context to generate answer."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


# ==============================================
#        Langchain Documents Generation
# ==============================================
def get_docs_from_pdf(file: str):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents


def get_docs_from_json(file: str):
    loader = JSONLoader(
        file_path=file, jq_schema=".messages[].content", text_content=False
    )
    documents = loader.load()
    return documents


# ==============================================
#               RAG Processing service
# ==============================================
def generate_answer_from_file(questions: list[str], documents: List[Document]):
    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Vector Embeddings
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )

    # Create Chain
    retriever = vectorstore.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run Batch Inference
    question_inputs = [{"input": question} for question in questions]
    results = rag_chain.batch(question_inputs)

    return results
