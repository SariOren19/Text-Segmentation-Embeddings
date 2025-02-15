# Text Segmentation & Embeddings

## Overview
This project processes **PDF and DOCX documents**, splits the text into chunks, generates embeddings, and stores them in **ChromaDB** for fast retrieval.

## Features
- Extracts text from PDFs & DOCX files.
- Splits text into **chunks** (overlapping words, sentence-based, or paragraph-based).
- Generates **embeddings** using Sentence Transformers.
- Stores and retrieves embeddings in **ChromaDB**.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SariOren19/Text-Segmentation-Embeddings.git
   cd Text-Segmentation-Embeddings

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. Run the script:
   ```bash
   python main.py

## Usage
- Edit file_path in main.py to specify your document.
- Choose a chunking method: 'overlap_word', 'sentence', or 'paragraph'.
- Choose a query. 
- Run the script for results.
- Notes:    The embeddings and data will be stored in chroma_db/.

## Example Query
   ```bash 
   query = "What is Supervised learning?"
   results = search_chromadb(query, top_k=3)
   print(results)