from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import re
from ..config import Config
from ..utils.pdf_utils import extract_text_from_pdf
from io import BytesIO
from .memory_service import MemoryService

llm = ChatGroq(model=Config.LLM_MODEL, temperature=0.2, api_key=Config.GROQ_API_KEY)
CANNOT_ANSWER_TEXT = "I cannot answer this from the uploaded document. Please ask a question grounded in the document content."


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    classification: str  # "related" or "off_topic"
    retrieved_docs: List[str]
    retrieval_scores: List[float]
    grade: str  # "relevant", "needs_refine", "irrelevant"
    response: str
    confidence_score: float
    history: List[dict]  # Added for memory: [{"role": "user", "content": "..."}, ...]


def _extract_label(raw_text: str, allowed_labels: List[str], default_label: str) -> str:
    text = (raw_text or "").strip().lower()
    for label in allowed_labels:
        if label in text:
            return label
    return default_label


def _history_to_text(history: List[dict], max_turns: int = 8) -> str:
    recent_history = history[-max_turns:] if history else []
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])


def _answer_style_instructions(question: str) -> str:
    q = (question or "").lower()
    instructions = [
        "Return only the final answer. Do not add prefaces like 'Here is...' or reasoning."
    ]

    if "bullet" in q or "points" in q or "list" in q:
        instructions.append(
            "Return 5-10 concise bullet points only, each starting with '- '. Focus on major topics, not exhaustive keywords."
        )

    line_match = re.search(r"(\d+)\s*line", q)
    if line_match:
        line_count = max(1, min(8, int(line_match.group(1))))
        instructions.append(
            f"Return exactly {line_count} lines, one sentence per line, plain text without numbering."
        )
    elif "summary" in q or "summarize" in q:
        instructions.append("Return a concise summary in 3-5 sentences.")
    else:
        instructions.append("Keep the answer concise: 1-3 short sentences unless the user asked for a list.")

    return "\n".join(instructions)


def _distance_to_confidence(distance: float) -> float:
    # Convert FAISS L2 distance into a bounded confidence heuristic.
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, distance))))


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _is_document_intent(question: str) -> bool:
    q = (question or "").lower()
    intent_keywords = [
        "document", "doc", "summary", "summarize", "overview", "about",
        "name", "skills", "experience", "mentioned", "profile", "who", "what"
    ]
    return any(keyword in q for keyword in intent_keywords)


def _is_summary_request(question: str) -> bool:
    q = (question or "").lower()
    return "summary" in q or "summarize" in q or "overview" in q


def _is_bullet_request(question: str) -> bool:
    q = (question or "").lower()
    return "bullet" in q or "points" in q or "list" in q


def _extract_requested_line_count(question: str) -> int | None:
    match = re.search(r"(\d+)\s*line", (question or "").lower())
    if not match:
        return None
    return max(1, min(10, int(match.group(1))))


def _format_summary_lines(summary_text: str, bullets_text: str, line_count: int) -> str:
    bullet_lines = []
    for line in (bullets_text or "").splitlines():
        cleaned = line.replace("- ", "", 1).strip()
        if cleaned:
            bullet_lines.append(cleaned)

    if bullet_lines:
        chosen = bullet_lines[:line_count]
        if len(chosen) < line_count:
            chosen.extend([""] * (line_count - len(chosen)))
        return "\n".join([line for line in chosen if line][:line_count])

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (summary_text or "").strip()) if s.strip()]
    if not sentences:
        return CANNOT_ANSWER_TEXT
    chosen = sentences[:line_count]
    if len(chosen) < line_count:
        chosen.extend([sentences[-1]] * (line_count - len(chosen)))
    return "\n".join(chosen[:line_count])


def _compute_confidence_score(question: str, docs: List[str], retrieval_scores: List[float], grade: str, response: str) -> float:
    if not docs or not retrieval_scores:
        return 0.2
    if response == CANNOT_ANSWER_TEXT:
        return 0.2

    # Retrieval confidence from top results.
    top_conf = _distance_to_confidence(retrieval_scores[0])
    top3 = retrieval_scores[:3]
    mean_top_conf = sum(_distance_to_confidence(score) for score in top3) / len(top3)

    # Lexical support confidence: how much question vocabulary appears in retrieved context.
    question_tokens = _tokenize(question)
    docs_tokens = _tokenize(" ".join(docs[:3]))
    if question_tokens:
        overlap_conf = len(question_tokens.intersection(docs_tokens)) / len(question_tokens)
    else:
        overlap_conf = 0.0

    normalized_grade = (grade or "").strip().lower()
    grade_factor = {
        "relevant": 1.0,
        "needs_refine": 0.65,
        "irrelevant": 0.25
    }.get(normalized_grade, 0.4)

    # Doc-level intent should not be over-penalized for short wording.
    if _is_document_intent(question):
        overlap_conf = max(overlap_conf, 0.45)
        grade_factor = max(grade_factor, 0.75 if grade == "needs_refine" else grade_factor)

    # Weighted blend with conservative scaling to avoid inflated high confidence.
    raw = (0.45 * top_conf) + (0.25 * mean_top_conf) + (0.30 * overlap_conf)
    score = raw * grade_factor

    # Calibration floor: if retriever+grader agree and answer is grounded, avoid misleadingly low confidence.
    if normalized_grade == "relevant" and response != CANNOT_ANSWER_TEXT:
        if _is_document_intent(question):
            score = max(score, 0.62)
        else:
            score = max(score, 0.55)

    # Refine path can still produce usable grounded answers; keep it at least medium-low.
    if normalized_grade == "needs_refine" and response != CANNOT_ANSWER_TEXT and _is_document_intent(question):
        score = max(score, 0.5)

    return max(0.0, min(1.0, score))


def _confidence_label(confidence_score: float) -> str:
    if confidence_score >= 0.82:
        return "high"
    if confidence_score >= 0.5:
        return "medium"
    return "low"


# Nodes
def rewrite_question(state: GraphState) -> GraphState:
    history_str = _history_to_text(state["history"])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template=(
                                "You rewrite user questions for document retrieval.\n"
                                "Keep the meaning exactly the same and produce one concise rewritten question only.\n"
                                "History:\n{history}\n"
                                "User question: {question}"
                            ))
    chain = LLMChain(llm=llm, prompt=prompt)
    rewritten = chain.run(question=state["question"], history=history_str)
    return {"rewritten_question": rewritten}


def classify_question(state: GraphState) -> GraphState:
    history_str = _history_to_text(state["history"])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template=(
                                "Classify whether the user question should be answered from the uploaded document.\n"
                                "Return only one label: related or off_topic.\n"
                                "History:\n{history}\n"
                                "Question: {question}"
                            ))
    chain = LLMChain(llm=llm, prompt=prompt)
    classification = chain.run(question=state["rewritten_question"], history=history_str)
    label = _extract_label(classification, ["related", "off_topic"], "off_topic")

    # Heuristic override: common document questions should stay in RAG flow.
    if label == "off_topic" and _is_document_intent(state["rewritten_question"]):
        label = "related"
    return {"classification": label}


def retrieve(state: GraphState) -> GraphState:
    if 'vectorstore' not in globals() or globals()['vectorstore'] is None:
        return {"retrieved_docs": [], "retrieval_scores": [], "grade": "irrelevant", "confidence_score": 0.0}
    docs_with_scores = globals()['vectorstore'].similarity_search_with_score(state["rewritten_question"], k=5)
    if not docs_with_scores:
        return {"retrieved_docs": [], "retrieval_scores": [], "confidence_score": 0.0}

    retrieved = [doc.page_content for doc, _ in docs_with_scores]
    distances = [float(score) for _, score in docs_with_scores]
    top_confidence = _distance_to_confidence(distances[0])
    return {"retrieved_docs": retrieved, "retrieval_scores": distances, "confidence_score": top_confidence}


def grade_retrieval(state: GraphState) -> GraphState:
    if not state["retrieved_docs"]:
        return {"grade": "irrelevant"}
    docs_str = "\n".join(state["retrieved_docs"])
    history_str = _history_to_text(state["history"])
    prompt = PromptTemplate(input_variables=["question", "docs", "history"],
                            template=(
                                "You are checking whether retrieved passages can answer the question.\n"
                                "Return only one label: relevant, needs_refine, or irrelevant.\n"
                                "- relevant: docs directly contain answer evidence.\n"
                                "- needs_refine: topic seems related but evidence is weak/indirect.\n"
                                "- irrelevant: docs do not support answering.\n"
                                "History:\n{history}\n"
                                "Question: {question}\n"
                                "Docs:\n{docs}"
                            ))
    chain = LLMChain(llm=llm, prompt=prompt)
    grade = chain.run(question=state["rewritten_question"], docs=docs_str, history=history_str)
    return {"grade": _extract_label(grade, ["relevant", "needs_refine", "irrelevant"], "irrelevant")}


def generate_answer(state: GraphState) -> GraphState:
    if not state["retrieved_docs"]:
        return {"response": CANNOT_ANSWER_TEXT}

    docs_str = "\n".join(state["retrieved_docs"])
    history_str = _history_to_text(state["history"])
    style_instructions = _answer_style_instructions(state["question"])
    prompt = PromptTemplate(input_variables=["question", "docs", "history"],
                            template=(
                                "You are a strict document-grounded assistant.\n"
                                "Rules:\n"
                                "1) Answer only from the provided docs.\n"
                                "2) Do not invent names, facts, dates, or roles.\n"
                                "3) If evidence is missing/unclear, reply exactly with:\n"
                                f"{CANNOT_ANSWER_TEXT}\n"
                                "4) Keep answer concise and directly relevant (max 2-3 short sentences).\n"
                                "5) Do not include internal reasoning, rewrite notes, or assumptions.\n"
                                "6) Follow the required output format exactly.\n"
                                "Output format instructions:\n"
                                f"{style_instructions}\n"
                                "History:\n{history}\n"
                                "Question: {question}\n"
                                "Docs:\n{docs}"
                            ))
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(question=state["rewritten_question"], docs=docs_str, history=history_str)
    answer = (answer or "").strip()
    if not answer:
        return {"response": CANNOT_ANSWER_TEXT}
    return {"response": answer}


def refine_question(state: GraphState) -> GraphState:
    history_str = _history_to_text(state["history"])
    prompt = PromptTemplate(input_variables=["question", "history"],
                            template=(
                                "Rewrite the question to be more specific for document retrieval.\n"
                                "Return only one refined question.\n"
                                "History:\n{history}\n"
                                "Question: {question}"
                            ))
    chain = LLMChain(llm=llm, prompt=prompt)
    refined = chain.run(question=state["rewritten_question"], history=history_str)
    state["rewritten_question"] = refined
    state = retrieve(state)
    state.update(grade_retrieval(state))
    if state.get("grade") == "relevant":
        state.update(generate_answer(state))
    else:
        state.update(cannot_answer(state))
    return state


def off_topic_response(state: GraphState) -> GraphState:
    if _is_document_intent(state.get("question", "")):
        return {
            "response": CANNOT_ANSWER_TEXT,
            "confidence_score": 0.2
        }
    return {
        "response": "This question appears off-topic for the uploaded document. Please ask about the document content.",
        "confidence_score": 0.2
    }


def cannot_answer(state: GraphState) -> GraphState:
    return {"response": CANNOT_ANSWER_TEXT, "confidence_score": 0.2}


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
upload_summary = ""
upload_bullets = ""
graph = build_graph()
memory = MemoryService()


def process_upload(file_content: BytesIO, filename: str):
    global vectorstore, full_text, upload_summary, upload_bullets
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
    upload_summary = summary
    upload_bullets = bullets

    # Clear history on new upload? Or keep? For now, keep as is; user can add clear endpoint if needed.
    return {"summary": summary, "bullets": bullets}


def process_chat(question: str):
    if vectorstore is None:
        raise ValueError("No file uploaded yet")

    # Deterministic path for summary/bullets to reduce hallucinations on common requests.
    if _is_summary_request(question) or _is_bullet_request(question):
        if not upload_summary and not upload_bullets:
            response = CANNOT_ANSWER_TEXT
            confidence_score = 0.2
            confidence_label = "low"
        else:
            if _is_bullet_request(question):
                response = upload_bullets or upload_summary
            else:
                line_count = _extract_requested_line_count(question)
                if line_count:
                    response = _format_summary_lines(upload_summary, upload_bullets, line_count)
                else:
                    response = upload_summary
            confidence_score = 0.9
            confidence_label = "high"

        memory.add_message("user", question)
        memory.add_message("assistant", response)
        return {
            "response": response,
            "confidence_score": round(confidence_score, 2),
            "confidence_label": confidence_label
        }

    inputs = {"question": question, "history": memory.get_history()}
    try:
        result = graph.invoke(inputs)
        response = result.get("response", CANNOT_ANSWER_TEXT)
        confidence_score = _compute_confidence_score(
            question=question,
            docs=result.get("retrieved_docs", []),
            retrieval_scores=result.get("retrieval_scores", []),
            grade=result.get("grade", "irrelevant"),
            response=response
        )
    except Exception:
        response = CANNOT_ANSWER_TEXT
        confidence_score = 0.2

    if response == CANNOT_ANSWER_TEXT:
        confidence_score = min(confidence_score, 0.2)

    confidence_score = max(0.0, min(1.0, confidence_score))
    confidence_label = _confidence_label(confidence_score)

    # Add to memory
    memory.add_message("user", question)
    memory.add_message("assistant", response)

    return {
        "response": response,
        "confidence_score": round(confidence_score, 2),
        "confidence_label": confidence_label
    }