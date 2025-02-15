import pdfplumber
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import chromadb

# Download NLTK tokenizer - used to split text into sentences and words.
nltk.download("punkt_tab")

# Load embedding model (converts text chunks into embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Initializes ChromaDB as a persistent storage database to store and retrieve embeddings.
collection = chroma_client.get_or_create_collection(name="documents") # creates or loads a collection named "documents" in ChromaDB.

# Extract text from documents
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Only pdf or docx files are supported")
    return text

# Text Splitting Methods
# Method 1: Overlapping Word Chunking
def chunk_text_overlapping_words(text, chunk_size=50, overlap=10):
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Method 2: Sentence-based splitting (every 4 sentences)
def chunk_text_by_sentence(text, max_sentences=4):
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]

# Method 3: Paragraph-based splitting
def chunk_text_by_paragraph(text):
    paragraphs = text.split("\n\n")  # Assumes paragraphs are separated by double newlines
    return [para.strip() for para in paragraphs if para.strip()]  # Remove empty paragraphs

# Convert text chunks into numerical embeddings
def get_embedding(text):
    return model.encode(text).tolist()

# Store embeddings in ChromaDB
def store_in_chromadb(embeddings, texts):
    for i, emb in enumerate(embeddings):
        collection.add(
            ids=[str(i)],  # Unique ID for each document
            embeddings=[emb],  # Store the numerical embedding
            documents=[texts[i]]  # Store the original text chunk
        )

# Search for relevant information in ChromaDB
def search_chromadb(query, top_k=3):
    query_embedding = get_embedding(query)  # Convert query to embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)  # Retrieve top results
    return results["documents"]

# Main Function to Process a Document
def process_document(file_path, method="sentence"):
    text = extract_text(file_path)  # Step 1: Extract text

    # Step 2: Choose chunking method
    if method == "overlap_word":
        chunks = chunk_text_overlapping_words(text)
    elif method == "sentence":
        chunks = chunk_text_by_sentence(text)
    elif method == "paragraph":
        chunks = chunk_text_by_paragraph(text)
    else:
        raise ValueError("Invalid chunking method. Choose 'overlap_word', 'sentence', or 'paragraph'.")

    # Step 3: Generate embeddings
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Step 4: Store embeddings in ChromaDB
    store_in_chromadb(embeddings, chunks)

    return chunks  # Return stored text chunks

# Running the Code
file_path = r"" # Enter file's path
chunk_method = "overlap_word" # Chose chunk method: 'overlap_word', 'sentence', or 'paragraph'

stored_chunks = process_document(file_path, method=chunk_method)
print(f"Document processed using {chunk_method}-based chunking and stored in ChromaDB!")

# Searching for Information
query = "" # Enter a query
results = search_chromadb(query, top_k=3)

print("Top Matching Results:\n")
for i, res in enumerate(results[0]):  # ChromaDB returns results in a nested list
    print(f"{i+1}. {res}\n")