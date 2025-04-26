from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from ..config import Config
from ..utils.pdf_utils import extract_text_from_pdf
from io import BytesIO
from .memory_service import MemoryService

llm = ChatGroq(model=Config.LLM_MODEL, temperature=0.7, api_key=Config.GROQ_API_KEY)


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    classification: str  # "related" or "off_topic"
    retrieved_docs: List[str]
    grade: str  # "relevant", "needs_refine", "irrelevant"
    response: str
    history: List[dict]  # Added for memory: [{"role": "user", "content": "..."}, ...]


# Nodes
def rewrite_question(state: GraphState) -> GraphState:
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"]])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template="Given chat history:\n{history}\nRewrite this question for better document retrieval: {question}")
    chain = LLMChain(llm=llm, prompt=prompt)
    rewritten = chain.run(question=state["question"], history=history_str)
    return {"rewritten_question": rewritten}


def classify_question(state: GraphState) -> GraphState:
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"]])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template="Given history:\n{history}\nClassify if this question is related to the uploaded document. Respond only with 'related' or 'off_topic': {question}")
    chain = LLMChain(llm=llm, prompt=prompt)
    classification = chain.run(question=state["rewritten_question"], history=history_str)
    return {"classification": classification.strip().lower()}


def retrieve(state: GraphState) -> GraphState:
    if 'vectorstore' not in globals() or globals()['vectorstore'] is None:
        return {"retrieved_docs": [], "grade": "irrelevant"}
    docs = globals()['vectorstore'].similarity_search(state["rewritten_question"], k=5)
    retrieved = [doc.page_content for doc in docs]
    return {"retrieved_docs": retrieved}


def grade_retrieval(state: GraphState) -> GraphState:
    if not state["retrieved_docs"]:
        return {"grade": "irrelevant"}
    docs_str = "\n".join(state["retrieved_docs"])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"]])
    prompt = PromptTemplate(input_variables=["question", "docs", "history"],
                            template="Given history:\n{history}\nGrade if these docs are relevant to the question. Respond only with 'relevant', 'needs_refine', or 'irrelevant': Question: {question}\nDocs: {docs}")
    chain = LLMChain(llm=llm, prompt=prompt)
    grade = chain.run(question=state["rewritten_question"], docs=docs_str, history=history_str)
    return {"grade": grade.strip().lower()}


def generate_answer(state: GraphState) -> GraphState:
    docs_str = "\n".join(state["retrieved_docs"])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"]])
    prompt = PromptTemplate(input_variables=["question", "docs", "history"],
                            template="Given history:\n{history}\nAnswer the question based on these docs: Question: {question}\nDocs: {docs}")
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(question=state["rewritten_question"], docs=docs_str, history=history_str)
    return {"response": answer}


def refine_question(state: GraphState) -> GraphState:
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["history"]])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template="Given history:\n{history}\nRefine this question to make it more specific for the document: {question}")
    chain = LLMChain(llm=llm, prompt=prompt)
    refined = chain.run(question=state["rewritten_question"], history=history_str)
    state["rewritten_question"] = refined
    state = retrieve(state)
    state = generate_answer(state)
    return state


def off_topic_response(state: GraphState) -> GraphState:
    return {"response": "This question seems off-topic from the uploaded document. Please ask about the content."}


def cannot_answer(state: GraphState) -> GraphState:
    return {"response": "Cannot answer based on the available information in the document."}


# Build graph
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("classify", classify_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_retrieval)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("refine", refine_question)
    workflow.add_node("off_topic", off_topic_response)
    workflow.add_node("cannot", cannot_answer)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "classify")
    workflow.add_conditional_edges(
        "classify",
        lambda state: state["classification"],
        {"related": "retrieve", "off_topic": "off_topic"}
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        lambda state: state["grade"],
        {"relevant": "generate", "needs_refine": "refine", "irrelevant": "cannot"}
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("refine", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("cannot", END)

    return workflow.compile()


# Global vars (for single instance; scale by making session-based)
vectorstore = None
full_text = ""
graph = build_graph()
memory = MemoryService()


def process_upload(file_content: BytesIO, filename: str):
    global vectorstore, full_text
    if filename.endswith('.pdf'):
        full_text = extract_text_from_pdf(file_content)
    else:
        full_text = file_content.getvalue().decode('utf-8')

    splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunks = splitter.split_text(full_text)


    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

    vectorstore = FAISS.from_texts(chunks, embeddings)

    summary_prompt = PromptTemplate(input_variables=["text"],
                                    template="Provide a short, concise paragraph summarizing the main ideas of this text without any headings or extra phrases: {text}")
    bullet_prompt = PromptTemplate(input_variables=["text"],
                                   template="Extract 5-10 key points from this text as a simple bullet list starting each with -, one per line, without any introduction, conclusion, or additional text: {text}")
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    bullet_chain = LLMChain(llm=llm, prompt=bullet_prompt)
    summary = summary_chain.run(text=full_text[:4000]).strip()
    bullets = bullet_chain.run(text=full_text[:4000]).strip()

    # Clear history on new upload? Or keep? For now, keep as is; user can add clear endpoint if needed.
    return {"summary": summary, "bullets": bullets}


def process_chat(question: str):
    if vectorstore is None:
        raise ValueError("No file uploaded yet")

    inputs = {"question": question, "history": memory.get_history()}
    result = graph.invoke(inputs)
    response = result["response"]

    # Add to memory
    memory.add_message("user", question)
    memory.add_message("assistant", response)

    return {"response": response}