
import streamlit as st
import hashlib

from rag import (
    extract_text_from_pdf,
    split_into_chunks,
    create_vector_database,
    search_similar_chunks,
    generate_answer
)

from db import (
    create_tables,
    save_document,
    save_chat,
    get_chat_history
)

st.set_page_config(
    page_title="NoteMind",
    page_icon="📚",
    layout="wide"
)

# Create database tables
create_tables()

st.title("📚 NoteMind")
st.subheader("Chat With Your Notes using AI")

st.write("Upload your PDF notes to get started.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Identify uploaded PDF
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    # Process only when a new PDF is uploaded
    if st.session_state.get("file_id") != file_id:

        with st.spinner("Processing PDF..."):

            try:
                uploaded_file.seek(0)

                text = extract_text_from_pdf(uploaded_file)
                chunks = split_into_chunks(text)

                if chunks:

                    index = create_vector_database(chunks)

                    # Save document in PostgreSQL
                    document_id = save_document(
                        uploaded_file.name
                    )

                    # Store processed data in session
                    st.session_state.file_id = file_id
                    st.session_state.document_id = document_id
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.text = text

                    st.success("PDF processed successfully!")

                else:
                    st.error("No readable text found in PDF.")

            except Exception as e:
                st.error(f"PDF processing error: {e}")

    # Continue only if PDF processing succeeded
    if st.session_state.get("file_id") == file_id:

        document_id = st.session_state.document_id
        chunks = st.session_state.chunks
        index = st.session_state.index
        text = st.session_state.text

        st.divider()
        st.subheader("💬 Ask Your Notes")

        # Question form
        with st.form("question_form"):

            question = st.text_input(
                "Enter your question"
            )

            submitted = st.form_submit_button(
                "Ask NoteMind"
            )

        latest_results = None

        if submitted and question.strip():

            with st.spinner("Finding relevant information..."):

                try:
                    results = search_similar_chunks(
                        question,
                        chunks,
                        index
                    )

                    with st.spinner("Generating AI answer..."):

                        answer = generate_answer(
                            question,
                            results
                        )

                    # Save question and answer in PostgreSQL
                    save_chat(
                        document_id,
                        question,
                        answer
                    )

                    latest_results = results

                    st.success("Answer generated and saved!")

                except Exception as e:
                    st.error(f"Error: {e}")

        # Display chat history
        st.divider()
        st.subheader("🕘 Chat History")

        history = get_chat_history(document_id)

        if history:

            for old_question, old_answer, created_at in history:

                with st.chat_message("user"):
                    st.write(old_question)

                with st.chat_message("assistant"):
                    st.write(old_answer)
                    st.caption(str(created_at))

        else:
            st.info("No conversations yet. Ask your first question!")

        # Display sources for latest question
        if latest_results:

            with st.expander("View Retrieved Sources"):

                for i, result in enumerate(latest_results):

                    st.write(f"**Source {i+1}**")

                    st.write(
                        "Similarity Score:",
                        round(result["score"], 3)
                    )

                    st.info(result["text"])

        # PDF statistics
        st.divider()
        st.subheader("📊 Document Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Words",
            len(text.split())
        )

        col2.metric(
            "Total Chunks",
            len(chunks)
        )

        col3.metric(
            "Stored Vectors",
            index.ntotal
        )

        with st.expander("View Extracted Text"):
            st.write(text[:3000])

        with st.expander("View First Chunk"):
            st.write(chunks[0])
