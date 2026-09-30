import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Get Gemini API key from Streamlit Secrets or .env
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
st.caption("Advanced RAG-powered document Q&A")


PDF_PATH = "terms_Recruitment_and_Trade_Policy_Manual.pdf"
INDEX_PATH = "faiss_index"


def process_documents(pdf_path):
    """Load PDF, split it into chunks, create embeddings, and save FAISS index."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(INDEX_PATH)

    return vector_store


def get_conversation_chain():
    """Create the Gemini + FAISS RAG chain."""

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the information provided
        in the document context.

        If the answer cannot be found in the provided context,
        clearly say that the document does not provide enough
        information. Do not invent facts.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


def main():

    # Check whether the PDF exists
    if not os.path.exists(PDF_PATH):
        st.error(
            f"PDF file not found: {PDF_PATH}"
        )

        st.info(
            "Upload the PDF to the same GitHub repository "
            "folder as app.py."
        )

        st.stop()

    # Create FAISS index the first time the app runs
    if not os.path.exists(INDEX_PATH):

        with st.spinner(
            "Processing PDF and generating embeddings..."
        ):
            process_documents(PDF_PATH)

        st.success(
            "Document processed successfully."
        )

    # Create RAG chain
    rag_chain = get_conversation_chain()

    # User question
    user_question = st.text_input(
        "Ask a question about the document:"
    )

    if user_question:

        with st.spinner("Searching the document..."):

            response = rag_chain.invoke(
                {
                    "input": user_question
                }
            )

        st.subheader("Answer")

        st.write(
            response["answer"]
        )


if __name__ == "__main__":
    main()
