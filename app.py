import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Get API key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Assistant")
st.caption("PDF Question Answering with RAG and Gemini")


PDF_PATH = "terms_Recruitment_and_Trade_Policy_Manual.pdf"
INDEX_PATH = "faiss_index"


@st.cache_resource
def create_vector_store(pdf_path):
    """Load PDF, split text, create embeddings and FAISS index."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def get_answer(vector_store, question):

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )
        st.stop()

    # Retrieve the 3 most relevant document chunks
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a document assistant.

Answer the user's question using ONLY the
information contained in the context below.

If the answer is not available in the context,
say:

"The document does not provide enough information
to answer this question."

Do not invent or assume information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    return response.content


def main():

    # Check PDF
    if not os.path.exists(PDF_PATH):

        st.error(
            f"PDF file not found: {PDF_PATH}"
        )

        st.info(
            "Upload the PDF to the same GitHub repository "
            "folder as app.py."
        )

        st.stop()

    # Create vector store
    with st.spinner(
        "Processing document and creating embeddings..."
    ):
        vector_store = create_vector_store(PDF_PATH)

    st.success("Document is ready.")

    # Question input
    question = st.text_input(
        "Ask a question about the document:"
    )

    if question:

        with st.spinner(
            "Searching the document..."
        ):

            answer = get_answer(
                vector_store,
                question
            )

        st.subheader("Answer")

        st.write(answer)


if __name__ == "__main__":
    main()
    
