"""
R2AI Stage 1 Test Pipeline
Runs RAG system (Hybrid + Reranking) on R2AIStage1DATA.json
Outputs results.json in required format
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import zipfile

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from main.chatbot import VietnameseLegalRAG
from config import Config
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)

class R2AITestPipeline:
    def __init__(self):
        """Initialize RAG system for testing"""
        print("🚀 Initializing R2AI Test Pipeline...")
        print(f"   📌 LLM Type: {'Local Qwen3-4B' if Config.USE_LOCAL_LLM else 'Gemini API'}")
        print(f"   📌 Embedding: {Config.EMBEDDING_MODEL}")
        print(f"   📌 Reranker: {Config.RERANKER_MODEL}")
        
        self.rag = VietnameseLegalRAG()
        if self.rag.bm25_retriever:
            self.rag.bm25_retriever.load_index()
            print("✅ BM25 index loaded successfully")
        self.results = []
        
    def load_test_data(self, filepath: str = "R2AIStage1DATA.json") -> List[Dict]:
        """Load test questions from R2AIStage1DATA.json"""
        print(f"\n📂 Loading test data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Test data file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} questions")
        return data
        
    def _format_law_title(self, law_id: str, raw_title: str) -> str:
        """Format strictly according to rules: Loại văn bản + Mã văn bản + Trích yếu"""
        return format_law_title(law_id, raw_title)
    
    def extract_relevant_info(self, retrieved_docs: List[Dict]) -> tuple:
        """
        Extract relevant_docs and relevant_articles from retrieved documents.

        Format:
        - relevant_docs:    ["mã văn bản|tên văn bản"]
        - relevant_articles:["mã văn bản|tên văn bản|Điều X"]
        """
        # Load law_id to title mapping
        mapping_path = Path(__file__).parent / "data" / "law_id_to_title.json"
        mapping = {}
        if mapping_path.exists():
            try:
                mapping = load_law_title_mapping(mapping_path)
            except Exception as e:
                print(f"⚠️ Warning loading law mapping: {e}")

        relevant_docs = []
        relevant_articles = []

        for doc in retrieved_docs:
            metadata  = doc.get('metadata', {})
            law_id    = metadata.get('law_id', '')
            article_id= metadata.get('article_id', '')
            article_title = metadata.get('title', '')

            if not law_id:
                continue

            # Look up clean human-readable title, fallback to law_id if not mapped.
            law_id = canonical_law_id(law_id)
            raw_title = get_mapping_title(mapping, law_id)
            law_title = self._format_law_title(law_id, raw_title)

            # --- relevant_docs: "law_id|law_title" ---
            doc_ref = f"{law_id}|{law_title}"
            if doc_ref not in relevant_docs:
                relevant_docs.append(doc_ref)

            # --- relevant_articles: "law_id|law_title|Điều X" ---
            label = article_label(article_title, article_id)
            if not label:
                continue

            article_ref = f"{law_id}|{law_title}|{label}"
            if article_ref not in relevant_articles:
                relevant_articles.append(article_ref)

        return relevant_docs, relevant_articles
    
    def process_question(self, question_id: int, question: str, timeout: int = 60) -> Dict:
        """
        Process single question through RAG system
        Using Hybrid + Reranking (best method)
        """
        try:
            start_time = time.time()
            
            # Retrieve documents using BEST method: Hybrid + Reranking
            retrieved_docs = self.rag.retrieve_documents(
                question,
                use_hybrid=True,
                use_reranking=True
            )
            
            # Get answer from RAG
            answer = self.rag.generate_answer(
                question,
                context=self.rag._build_context_from_documents(retrieved_docs),
                is_fallback=False
            )
            
            # Extract relevant information
            relevant_docs, relevant_articles = self.extract_relevant_info(retrieved_docs)
            
            elapsed_time = time.time() - start_time
            
            result = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "relevant_docs": relevant_docs,
                "relevant_articles": relevant_articles
            }

            print(f"   ✅ Q{question_id}: {len(retrieved_docs)} docs, {len(relevant_articles)} articles ({elapsed_time:.1f}s)")
            return result

        except Exception as e:
            print(f"\n❌ Error processing question {question_id}: {str(e)}")

            # Trả về đúng schema ngay cả khi lỗi (không thêm field "error")
            return {
                "id": question_id,
                "question": question,
                "answer": f"Lỗi xử lý câu hỏi: {str(e)}",
                "relevant_docs": [],
                "relevant_articles": []
            }
    
    def run_pipeline(self, test_data: List[Dict], output_file: str = "results.json") -> str:
        """
        Run full pipeline on all test questions
        """
        print(f"\n{'='*80}")
        print("🔄 RUNNING R2AI TEST PIPELINE (Hybrid + Reranking)")
        print(f"{'='*80}\n")
        
        total_questions = len(test_data)
        
        for item in tqdm(test_data, desc="Processing questions", total=total_questions):
            question_id = item['id']
            question = item['question']
            
            result = self.process_question(question_id, question)
            self.results.append(result)
            
            # Fix MPS memory leak
            import torch
            import gc
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save results
        print(f"\n💾 Saving results to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved: {output_file}")
        
        # Print summary
        self._print_summary(output_file)
        
        return output_file
    
    def _print_summary(self, output_file: str):
        """Print processing summary"""
        total = len(self.results)
        errors = sum(1 for r in self.results if 'error' in r)
        success = total - errors
        
        print(f"\n{'='*80}")
        print("📊 SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Processed: {success}/{total} questions successfully")
        print(f"❌ Failed: {errors}/{total} questions")
        print(f"📁 Output file: {output_file}")
        print(f"{'='*80}\n")
    
    def create_submission_zip(self, results_file: str = "results.json", 
                             output_zip: str = "submission.zip") -> str:
        """
        Create flat zip file with results.json (no subdirectories)
        """
        print(f"\n📦 Creating submission zip file...")
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        # Create flat zip (no directory structure)
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add only the filename, not the full path
            zf.write(results_file, arcname=results_file)
        
        zip_size = os.path.getsize(output_zip)
        print(f"✅ Zip file created: {output_zip}")
        print(f"   📊 Size: {zip_size / 1024 / 1024:.2f} MB")
        
        return output_zip


def main():
    """Main entry point"""
    try:
        # Initialize pipeline
        pipeline = R2AITestPipeline()
        
        # Load test data
        test_data = pipeline.load_test_data("R2AIStage1DATA.json")
        
        # Run pipeline
        results_file = pipeline.run_pipeline(test_data, "results.json")
        
        # Create submission zip
        zip_file = pipeline.create_submission_zip("results.json", "submission.zip")
        
        print(f"\n🎉 COMPLETED SUCCESSFULLY!")
        print(f"   📄 Results: {results_file}")
        print(f"   📦 Submission: {zip_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
