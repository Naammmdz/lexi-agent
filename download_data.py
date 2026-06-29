#!/usr/bin/env python3
"""
Download Zalo AI 2021 Legal Text Retrieval dataset from Kaggle
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def download_from_kaggle():
    """Download dataset from Kaggle using kaggle CLI"""
    print("📥 Downloading from Kaggle...")
    
    try:
        # Check if kaggle CLI is installed
        result = subprocess.run(["kaggle", "--version"], capture_output=True)
        if result.returncode != 0:
            print("❌ Kaggle CLI not found. Install with: pip install kaggle")
            return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Download dataset
    print("⏳ Downloading zalo-ai-2021 dataset (this may take a while)...")
    cmd = ["kaggle", "datasets", "download", "-d", "hariwh0/zaloai2021-legal-text-retrieval", "-p", "data", "--unzip"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr}")
        return False
    
    print("✅ Dataset downloaded successfully!")
    return True

def organize_data():
    """Organize downloaded data into expected structure"""
    print("\n📂 Organizing data files...")
    
    data_dir = Path("data")
    
    # Check corpus files and create legal_corpus.json if needed
    corpus_dir = data_dir / "corpus"
    if corpus_dir.exists():
        print(f"✅ Found corpus directory with {len(list(corpus_dir.glob('*.csv')))} CSV files")
        
        # Convert corpus CSVs to legal_corpus.json if needed
        if not (corpus_dir / "legal_corpus.json").exists():
            print("🔄 Creating legal_corpus.json from CSV files...")
            create_corpus_json(corpus_dir)
    else:
        print("❌ Corpus directory not found")
        return False
    
    # Check train data
    train_dir = data_dir / "train"
    if train_dir.exists():
        print(f"✅ Found train directory")
    
    # Check/create stopwords
    utils_dir = data_dir / "utils"
    os.makedirs(utils_dir, exist_ok=True)
    
    stopwords_file = utils_dir / "stopwords.txt"
    if not stopwords_file.exists():
        print("📝 Creating Vietnamese stopwords...")
        create_stopwords(stopwords_file)
    
    return True

def create_corpus_json(corpus_dir):
    """Create legal_corpus.json from CSV files"""
    import csv
    
    corpus_data = {}
    
    # Read from legal_corpus_original.csv or similar
    csv_files = [
        "legal_corpus_original.csv",
        "legal_corpus_merged_u369.csv",
        "legal_corpus.csv"
    ]
    
    for csv_file in csv_files:
        path = corpus_dir / csv_file
        if path.exists():
            print(f"  Reading {csv_file}...")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'id' in row and 'text' in row:
                        corpus_data[row['id']] = {
                            'id': row['id'],
                            'text': row['text'],
                            'title': row.get('title', ''),
                            'law_id': row.get('law_id', ''),
                            'article_id': row.get('article_id', '')
                        }
            break
    
    if corpus_data:
        output_file = corpus_dir / "legal_corpus.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(corpus_data.values()), f, ensure_ascii=False, indent=2)
        print(f"✅ Created legal_corpus.json with {len(corpus_data)} documents")

def create_stopwords(output_file):
    """Create Vietnamese stopwords file"""
    vietnamese_stopwords = [
        "và", "là", "cái", "có", "được", "của", "để", "từ", "với", "trong",
        "ở", "trên", "dưới", "sau", "trước", "giữa", "ngoài", "ngoài",
        "những", "các", "cách", "mà", "này", "kia", "đó", "đây",
        "như", "nếu", "nên", "nhưng", "hay", "mà", "vì", "sao", "thì", "tại sao",
        "tôi", "tao", "ta", "mình", "bạn", "anh", "chị", "cô", "ông", "bà",
        "không", "chưa", "chỉ", "mới", "thôi", "rồi", "còn", "lại", "đã", "sẽ"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vietnamese_stopwords))
    print(f"✅ Created Vietnamese stopwords ({len(vietnamese_stopwords)} words)")

def main():
    print("🏛️  Zalo AI Legal Data Download Script")
    print("="*50)
    
    # Step 1: Download
    if not download_from_kaggle():
        print("\n⚠️  Download failed. You can manually download from:")
        print("    https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval")
        print("\n    Extract to: data/")
        sys.exit(1)
    
    # Step 2: Organize
    if not organize_data():
        print("\n❌ Data organization failed")
        sys.exit(1)
    
    print("\n✅ Data setup completed!")
    print("\nNext steps:")
    print("  1. python3 setup_system.py           # Setup indices")
    print("  2. python3 test_r2ai_pipeline.py     # Run test pipeline")

if __name__ == "__main__":
    main()
