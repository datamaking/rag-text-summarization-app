import streamlit as st
import os
import tempfile
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import time

# Set page configuration
st.set_page_config(
    page_title="IT Support Knowledge Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0

# Set up the embeddings model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Set up the LLM
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN"),
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

# Function to load existing vector store
def load_existing_vector_store(persist_directory="./chroma_db"):
    # Check if the directory exists
    if not os.path.exists(persist_directory):
        return None, 0
    
    try:
        # Get the embeddings model
        embeddings = get_embeddings_model()
        
        # Load the existing vector store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Get the collection and count the documents
        collection = vector_store._collection
        document_count = collection.count()
        
        return vector_store, document_count
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None, 0

# Function to load and process documents
def process_document(file, file_name):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name
    
    # Load document based on file type
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(temp_file_path)
    elif file_name.endswith('.docx'):
        loader = Docx2txtLoader(temp_file_path)
    elif file_name.endswith('.txt'):
        loader = TextLoader(temp_file_path)
    elif file_name.endswith('.md'):
        loader = UnstructuredMarkdownLoader(temp_file_path)
    else:
        os.unlink(temp_file_path)
        return None, "Unsupported file format"
    
    # Load the document
    try:
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = file_name
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return chunks, None
    except Exception as e:
        os.unlink(temp_file_path)
        return None, str(e)

# Function to add documents to the vector store
def add_documents_to_vector_store(chunks):
    embeddings = get_embeddings_model()
    
    if st.session_state.vector_store is None:
        # Create a new vector store
        st.session_state.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    else:
        # Add to existing vector store
        st.session_state.vector_store.add_documents(chunks)
    
    # Persist changes to disk
    st.session_state.vector_store.persist()
    
    # Update document and chunk counts
    st.session_state.document_count += 1
    st.session_state.chunk_count += len(chunks)

# Function to query the vector store
def query_vector_store(query, top_k=5):
    if st.session_state.vector_store is None:
        return [], "No documents have been uploaded yet."
    
    # Create a retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    # Get relevant documents
    try:
        docs = retriever.get_relevant_documents(query)
        return docs, None
    except Exception as e:
        return [], str(e)

# Function to generate an answer using RAG
def generate_answer(query, docs):
    llm = get_llm()
    
    # Create a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=True
    )
    
    # Generate answer
    try:
        result = qa_chain({"query": query})
        return result["result"], result["source_documents"], None
    except Exception as e:
        return "", [], str(e)

# Function to summarize text
def summarize_text(text, length_factor=0.5, style="technical"):
    llm = get_llm()
    
    # Create a document from the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.create_documents([text])
    
    # Create a summarization chain based on style
    if style == "bullet":
        prompt_template = """
        Summarize the following text in bullet points:
        
        {text}
        
        BULLET POINT SUMMARY:
        """
    elif style == "simplified":
        prompt_template = """
        Summarize the following technical text in simple, non-technical language:
        
        {text}
        
        SIMPLIFIED SUMMARY:
        """
    else:  # technical
        prompt_template = """
        Provide a technical summary of the following text, maintaining technical accuracy:
        
        {text}
        
        TECHNICAL SUMMARY:
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    # Create the chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt
    )
    
    # Generate summary
    try:
        summary = chain.run(docs)
        return summary, None
    except Exception as e:
        return "", str(e)

# Main application UI
def main():
    st.title("IT Support Knowledge Assistant")
    st.markdown("Upload technical documents, query your knowledge base, and generate summaries")
    
    # Load existing vector store if it exists
    if st.session_state.vector_store is None:
        with st.spinner("Loading existing knowledge base..."):
            vector_store, doc_count = load_existing_vector_store()
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                # Estimate chunk count (this is approximate)
                st.session_state.document_count = doc_count
                st.session_state.chunk_count = doc_count * 5  # Rough estimate
                st.success(f"Loaded existing knowledge base with approximately {doc_count} documents")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "🔍 Query Knowledge Base", "📝 Summarize Text"])
    
    # Tab 1: Upload Documents
    with tab1:
        st.header("Upload Technical Documents")
        st.markdown("Add documents to your knowledge base for future queries")
        
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    chunks, error = process_document(file, file.name)
                    
                    if error:
                        st.error(f"Error processing {file.name}: {error}")
                    else:
                        add_documents_to_vector_store(chunks)
                        st.session_state.uploaded_files.append(file.name)
                        status_text.text(f"Added {file.name} to knowledge base ({len(chunks)} chunks)")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("All documents processed!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"Added {len(uploaded_files)} documents to your knowledge base")
        
        # Display knowledge base status
        st.subheader("Knowledge Base Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.document_count)
        with col2:
            st.metric("Text Chunks", st.session_state.chunk_count)
        
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Documents")
            for file in st.session_state.uploaded_files[-5:]:
                st.text(f"✅ {file}")
            
            if len(st.session_state.uploaded_files) > 5:
                st.text(f"... and {len(st.session_state.uploaded_files) - 5} more")
    
    # Tab 2: Query Knowledge Base
    with tab2:
        st.header("Query Your Knowledge Base")
        st.markdown("Ask questions and get answers based on your uploaded documents")
        
        query = st.text_area("Enter your query", height=100, placeholder="Describe the issue or ask a question about your technical documentation...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
        with col2:
            search_button = st.button("Search Knowledge Base", type="primary", disabled=not query)
        
        if search_button and query:
            with st.spinner("Searching knowledge base..."):
                # Get relevant documents
                docs, error = query_vector_store(query, top_k)
                
                if error:
                    st.error(f"Error: {error}")
                elif not docs:
                    st.warning("No relevant documents found in the knowledge base.")
                else:
                    # Generate answer
                    with st.spinner("Generating answer..."):
                        answer, source_docs, error = generate_answer(query, docs)
                        
                        if error:
                            st.error(f"Error generating answer: {error}")
                        else:
                            # Display answer
                            st.subheader("Answer")
                            st.write(answer)
                            
                            # Display sources
                            st.subheader("Sources")
                            for i, doc in enumerate(source_docs[:top_k]):
                                with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                                    st.markdown(f"**Relevance Score**: {doc.metadata.get('score', 'N/A')}")
                                    st.markdown(f"**Content**: {doc.page_content[:500]}...")
    
    # Tab 3: Summarize Text
    with tab3:
        st.header("Summarize Technical Text")
        st.markdown("Generate concise summaries of technical documents or incident reports")
        
        text_to_summarize = st.text_area("Enter text to summarize", height=200, placeholder="Paste the technical text you want to summarize...")
        
        col1, col2 = st.columns(2)
        with col1:
            length_factor = st.slider("Summary Length", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        with col2:
            style = st.selectbox("Summary Style", options=["technical", "simplified", "bullet"])
        
        if st.button("Generate Summary", type="primary", disabled=not text_to_summarize):
            with st.spinner("Generating summary..."):
                summary, error = summarize_text(text_to_summarize, length_factor, style)
                
                if error:
                    st.error(f"Error generating summary: {error}")
                else:
                    st.subheader("Summary")
                    st.write(summary)
                    
                    # Add copy button
                    if st.button("Copy to Clipboard"):
                        st.code(summary)
                        st.success("Summary copied to clipboard!")

if __name__ == "__main__":
    main()