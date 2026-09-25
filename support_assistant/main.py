from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
COLLECTION_NAME = "zepto_policies"
VECTOR_STORE_PATH = ROOT / "chroma_store"

PROMPT_TEMPLATE = """Answer the Zepto customer question using only the policy text below.
Keep the answer short and return JSON with these fields: answer, sources, confidence.
Do not add information that is not in the policy text.
Example:
User: "What is the delivery fee for orders below INR 149?"
Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Answer: {{"answer": "Orders below INR 149 incur a flat INR 25 delivery fee.", "sources": ["doc_01.txt"], "confidence": 0.97}}

User question: {query}
Context:
{context}
"""


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: float = Field(..., ge=0.0, le=1.0)


class AgentState(TypedDict):
    query: str
    intent: str
    top_chunks: list[dict[str, Any]]
    answer: str
    sources: list[str]
    confidence: float


class PolicyAssistant:
    def __init__(self) -> None:
        self.mock_llm = os.getenv("MOCK_LLM", "1") not in {"0", "False", "false"}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self._bootstrap_collection()

    def _bootstrap_collection(self) -> None:
        if self.collection.count() > 0:
            return
        docs: list[str] = []
        ids: list[str] = []
        for doc_path in sorted(DOCS_DIR.glob("*.txt")):
            docs.append(doc_path.read_text(encoding="utf-8").strip())
            ids.append(doc_path.name)
        self.collection.add(documents=docs, ids=ids)

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        policy_keywords = (
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours",
        )
        return "policy_question" if any(keyword in q for keyword in policy_keywords) else "general_question"

    def retrieve_chunks(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return [
            {"id": doc_id, "document": text}
            for doc_id, text in zip(results["ids"][0], results["documents"][0])
        ]

    def classify_intent_node(self, state: AgentState) -> AgentState:
        state["intent"] = self.classify_intent(state["query"])
        return state

    def route_decision(self, state: AgentState) -> str:
        return state["intent"]

    def direct_answer_node(self, state: AgentState) -> AgentState:
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
        return state

    def retrieve_and_answer_node(self, state: AgentState) -> AgentState:
        top_chunks = self.retrieve_chunks(state["query"], top_k=3)
        state["top_chunks"] = top_chunks
        state["sources"] = [chunk["id"] for chunk in top_chunks]

        if self.mock_llm:
            snippet = top_chunks[0]["document"][:200] if top_chunks else "the policy records"
            state["answer"] = f"According to the policy: {snippet}"
            state["confidence"] = 1.0
            return state

        context = "\n\n".join(chunk["document"] for chunk in top_chunks)
        prompt = PROMPT_TEMPLATE.format(query=state["query"], context=context)
        payload = self._parse_and_validate(self._call_real_llm(prompt), state["sources"])
        state["answer"] = payload["answer"]
        state["sources"] = payload["sources"]
        state["confidence"] = payload["confidence"]
        return state

    def _call_real_llm(self, prompt: str) -> str:
        return json.dumps({
            "answer": "The real language-model connection is not configured yet.",
            "sources": ["doc_01.txt"],
            "confidence": 0.9,
        })

    def _parse_and_validate(self, raw_output: str, fallback_sources: list[str]) -> dict[str, Any]:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        answer = str(payload.get("answer", "Unable to answer using the provided context."))
        sources = payload.get("sources")
        if not isinstance(sources, list):
            sources = fallback_sources
        confidence = max(0.0, min(float(payload.get("confidence", 0.5)), 1.0))
        return {"answer": answer, "sources": sources, "confidence": confidence}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer_node)
        workflow.add_node("direct_answer", self.direct_answer_node)
        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_decision,
            {
                "policy_question": "retrieve_and_answer",
                "general_question": "direct_answer",
            },
        )
        workflow.add_edge("retrieve_and_answer", END)
        workflow.add_edge("direct_answer", END)
        return workflow.compile()


assistant = PolicyAssistant()
app = FastAPI(title="Zepto Support Assistant")
app_graph = assistant.build_graph()


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: AskRequest) -> AnswerResponse:
    result = app_graph.invoke({
        "query": payload.query,
        "intent": "",
        "top_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    })
    return AnswerResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
