import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.assistant import PCBKnowledgeAssistant
from rag.loader import KnowledgeDocument, chunk_documents, discover_knowledge_files, load_knowledge_file
from web_interface import create_app


class RAGTests(unittest.TestCase):
    def test_plain_document_discovery_and_chunk_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            source = knowledge / "guide.md"
            source.write_text("# 数据规则\n\nOK用于训练，NG用于测试。" * 20, encoding="utf-8")
            files = discover_knowledge_files(knowledge, [])
            self.assertEqual(files, [source.resolve()])
            documents = load_knowledge_file(source, root)
            chunks = chunk_documents(documents, chunk_size=220, overlap=30)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0].metadata["citation"], "[knowledge/guide.md]")

    def test_evidence_mode_answers_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs" / "rag.yaml").write_text(
                "knowledge_dir: knowledge\npersist_dir: data/rag_index\ncollection_name: test\n"
                "embedding_model: test-model\nembedding_device: cpu\nchunk_size: 300\nchunk_overlap: 30\n",
                encoding="utf-8",
            )
            assistant = PCBKnowledgeAssistant(root)
            retrieval = {
                "results": [
                    {
                        "rank": 1,
                        "content": "来源：README.md\n\nOK训练，NG测试。",
                        "source": "README.md",
                        "page": None,
                        "citation": "[README.md]",
                        "distance": 0.2,
                    }
                ]
            }
            with patch("rag.assistant.search_knowledge", return_value=retrieval), patch.dict(
                "os.environ", {"RAG_LLM_MODEL": "", "RAG_LLM_API_KEY": ""}, clear=False
            ):
                result = assistant.ask("NG是否训练？")
            self.assertEqual(result["mode"], "evidence")
            self.assertIn("OK训练，NG测试", result["answer"])
            self.assertEqual(result["sources"][0]["citation"], "[README.md]")

    def test_chat_api(self) -> None:
        fake = type(
            "FakeAssistant",
            (),
            {
                "status": lambda self: {"ready": True, "mode": "evidence"},
                "ask": lambda self, question, history: {
                    "answer": f"回答：{question}",
                    "sources": [],
                    "mode": "evidence",
                    "warning": None,
                },
            },
        )()
        with patch("web_interface.get_rag_assistant", return_value=fake):
            client = create_app().test_client()
            response = client.post("/api/chat", json={"message": "如何训练？", "history": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"]["answer"], "回答：如何训练？")


if __name__ == "__main__":
    unittest.main()
