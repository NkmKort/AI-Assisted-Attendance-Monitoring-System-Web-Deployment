import streamlit as st
import pandas as pd
import os
import math
import nest_asyncio
import logging
import sys
from datetime import date as date_type
from dotenv import load_dotenv
import daily_report as dr
#kael
import langchain_google_vertexai

# Create a fake module path so old ragas code doesn't crash
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
#stp
# --- Core LangChain Imports ---
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Stable Ragas V1 Imports ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig

# --- Ragas / LangChain Verbose Debug Logging ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("ragas").setLevel(logging.DEBUG)
logging.getLogger("ragas.metrics").setLevel(logging.DEBUG)
logging.getLogger("ragas.llms").setLevel(logging.DEBUG)
logging.getLogger("langchain_groq").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

# Apply nested asyncio to prevent Streamlit threading conflicts with Ragas
nest_asyncio.apply()

# ==========================================
# 1. Page Configuration & Environment
# ==========================================
st.set_page_config(page_title="FateAutomata Kiosk", page_icon="👁️", layout="wide")
load_dotenv()

# ==========================================
# 2. Load RAG & Evaluation Components
# ==========================================
@st.cache_resource
def load_ai_components():
    """Loads all AI models, the FAISS database, and the Judge LLM once."""
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    
    vector_store = FAISS.load_local(
        "Knowledge Base", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Generator Model
    generator_llm = ChatCohere(model="command-a-03-2025")
    
    # Judge Model for Ragas (Llama 3.3 via Groq)
    judge_llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.0
        # model_kwargs={"response_format": {"type": "json_object"}} # Commented out to avoid the Groq JSON trap
    )
    
    return vector_store, generator_llm, judge_llm, embeddings

vector_store, generator_llm, judge_llm, embeddings_model = load_ai_components()

# ==========================================
# 3. User Interface Layout
# ==========================================
st.title("FateAutomata AI Attendance System")

tab1, tab2, tab3 = st.tabs(["📊 Live Attendance Dashboard", "🤖 Evaluated Handbook Chatbot", "📋 Daily Report Generator"])
# ------------------------------------------
# TAB 1: The Attendance Dashboard
# ------------------------------------------
with tab1:
    st.header("Live Attendance & Compliance Logs")
    CSV_FILE = "attendance_log.csv"
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
        
    df = dr.load_and_prepare()
    
    if not df.empty:
        # Safely convert string variations of 'True'/'False' into actual Pandas booleans
        # This prevents calculation errors when counting metrics
        for col in ['Cloud_Synced', 'Lanyard_Compliant', 'DressCode_Compliant']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes', 't'])
        
        # Expand metrics to utilize the new YOLOv8 data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Logs Recorded", len(df))
        with col2:
            synced = len(df[df['Cloud_Synced'] == True]) if not df.empty else 0
            st.metric("Pushed to Supabase", synced)
        with col3:
            if 'Lanyard_Compliant' in df.columns:
                l_violations = len(df[df['Lanyard_Compliant'] == False]) if not df.empty else 0
                st.metric("🚨 Missing IDs", l_violations)
        with col4:
            if 'DressCode_Compliant' in df.columns:
                d_violations = len(df[df['DressCode_Compliant'] == False]) if not df.empty else 0
                st.metric("👕 Dress Code Violations", d_violations)
            
        # Display the dataframe (newest logs at the top)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance logs found in Supabase.")

# ------------------------------------------
# TAB 2: The Closed-Corpus Chatbot with Ragas
# ------------------------------------------
with tab2:
    st.header("Mapúa Prefect of Discipline Assistant")
    st.markdown("Ask any question regarding university policies. My answers are mathematically evaluated for faithfulness to the handbook using Llama 3.3.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                st.caption(message["metrics"])

    if user_query := st.chat_input("E.g., What is the penalty for a 2nd offense of losing an ID?"):
        
        st.chat_message("user").markdown(user_query)
        
        # Retrieve Context
        docs = vector_store.similarity_search(user_query, k=4)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

        strict_prompt = f"""
        You are the official AI Assistant for the Mapúa University Student Handbook.
        
        CRITICAL RULE: You must answer the user's question STRICTLY using ONLY the information provided in the CONTEXT below. 
        If the answer cannot be found in the CONTEXT, you are forbidden from guessing or using your general training data. 
        Instead, you must reply exactly with: "I am sorry, but that information is not covered in the Mapúa Student Handbook."

        CONTEXT:
        {retrieved_context}

        USER QUESTION:
        {user_query}
        """

        with st.spinner("Generating response..."):
            response = generator_llm.invoke(strict_prompt).content
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # --- RAGAS EVALUATION ---
            fallback_phrase = "I am sorry, but that information is not covered in the Mapúa Student Handbook."

            if fallback_phrase in response:
                metrics_text = "📊 *Ragas Evaluation:* N/A (Out of Context / Successful Fallback)"
            else:
                try:
                    with st.spinner("Evaluating response with Llama 3.3 Judge..."):

                        data_sample = {
                            "question": [user_query],
                            "answer": [response],
                            "contexts": [[doc.page_content for doc in docs]]
                        }
                        dataset = Dataset.from_dict(data_sample)
                        
                        # Instantiate the local embeddings
                        hf_eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                        # THE CRITICAL FIX: Override Groq's n>1 limitation
                        answer_relevancy.strictness = 1

                        eval_result = evaluate(
                            dataset=dataset,
                            metrics=[faithfulness, answer_relevancy],
                            llm=judge_llm,
                            embeddings=hf_eval_embeddings,
                            run_config=RunConfig(max_workers=1)
                        )

                        df_result = eval_result.to_pandas()

                        f_score = float(df_result["faithfulness"].iloc[0]) if "faithfulness" in df_result.columns else float('nan')
                        r_score = float(df_result["answer_relevancy"].iloc[0]) if "answer_relevancy" in df_result.columns else float('nan')

                        f_display = "N/A" if math.isnan(f_score) else f"{f_score:.2f}"
                        r_display = "N/A" if math.isnan(r_score) else f"{r_score:.2f}"

                        metrics_text = f"📊 *Ragas Evaluation:* Faithfulness: {f_display} | Relevancy: {r_display}"

                except Exception as e:
                    metrics_text = f"📊 *Ragas Error:* {str(e)}"

            st.caption(metrics_text)

        # Save to session history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "metrics": metrics_text
        })

# ------------------------------------------
# TAB 3: Daily Report Generator
# ------------------------------------------
with tab3:
    st.header("Daily Compliance Report Generator")
    st.markdown(
        "Select a date, enter a recipient email, and click *Generate Report* "
        "to produce an AI-written formal report and email draft powered by Cohere Command A."
    )

    full_df = dr.load_and_prepare()

    if full_df.empty:
        st.warning("No attendance records found in Supabase. Check your connection or tables.")
        st.stop()

    # ── Row 1: Date picker + Email input ─────────────────────────────────
    col_date, col_email = st.columns([1, 2])

    with col_date:
        clean_dates = full_df["_date"].dropna().unique()
        available_dates = sorted([d for d in clean_dates if not pd.isna(d)], reverse=True)
        default_date = available_dates[0] if available_dates else date_type.today()
        selected_date = st.date_input(
            "Report Date",
            value=default_date,
            min_value=min(available_dates) if available_dates else date_type.today(),
            max_value=date_type.today(),
        )

    with col_email:
        recipient_email = st.text_input(
            "Recipient Email Address",
            placeholder="discipline.office@mapua.edu.ph",
        )

    # ── Filtered data for selected date ──────────────────────────────────
    daily_df = dr.filter_by_date(full_df, selected_date)
    stats = dr.build_stats(daily_df)

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students Logged", stats["total"])
    m2.metric("Fully Compliant", stats["compliant"])
    m3.metric("Missing ID", stats["lanyard_violations"])
    m4.metric("Dress Code Violations", stats["dress_violations"])

    # ── Filtered table ────────────────────────────────────────────────────
    if daily_df.empty:
        st.info(
            f"No attendance records found for *{selected_date}*. "
            "This may be a weekend, holiday, or a date before the system was deployed."
        )
    else:
        st.dataframe(daily_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Generate Report button ────────────────────────────────────────────
    generate_clicked = st.button("Generate Report", type="primary", disabled=daily_df.empty)

    if generate_clicked:
        # Validation
        if not recipient_email or "@" not in recipient_email:
            st.error("Please enter a valid recipient email address before generating the report.")
            st.stop()

        with st.spinner("Retrieving handbook policies and generating report via Cohere Command A..."):
            try:
                word_report, email_draft = dr.generate_report(
                    llm=generator_llm,
                    vector_store=vector_store,
                    selected_date=selected_date,
                    recipient_email=recipient_email,
                    daily_df=daily_df,
                )
                # Persist outputs in session state so they survive reruns
                st.session_state["report_word"] = word_report
                st.session_state["report_email"] = email_draft
                st.session_state["report_date"] = selected_date
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.stop()

    # ── Render outputs (persisted across reruns) ──────────────────────────
    if "report_word" in st.session_state:
        report_date = st.session_state["report_date"]

        with st.expander("📄 Part 1 — Formal Administrative Report (Word Document)", expanded=True):
            edited_report = st.text_area(
                label="Word Report",
                value=st.session_state["report_word"],
                height=500,
                key="report_word_editor",
                label_visibility="collapsed",
            )
            c_dl, c_sv = st.columns([1, 1])
            with c_dl:
                docx_bytes = dr.export_to_docx(edited_report, report_date)
                st.download_button(
                    label="⬇️ Download as .docx",
                    data=docx_bytes,
                    file_name=f"compliance_report_{report_date}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            with c_sv:
                if st.button("💾 Save Report Draft Changes", key="save_report_draft"):
                    st.session_state["report_word"] = edited_report
                    st.success("Report draft changes saved!")

        with st.expander("✉️ Part 2 — Email Draft", expanded=True):
            edited_email = st.text_area(
                label="Email Draft",
                value=st.session_state["report_email"],
                height=300,
                key="report_email_editor",
                label_visibility="collapsed",
            )
            if st.button("💾 Save Email Draft Changes", key="save_email_draft"):
                st.session_state["report_email"] = edited_email
                st.success("Email draft changes saved!")

            st.divider()

            send_clicked = st.button("Send Email Now", type="primary")
            if send_clicked:
                if not recipient_email or "@" not in recipient_email:
                    st.error("Enter a valid recipient email address at the top of the page first.")
                else:
                    docx_bytes = dr.export_to_docx(edited_report, report_date)
                    filename = f"compliance_report_{report_date}.docx"
                    subject = f"Daily Compliance Report — {report_date}"
                    with st.spinner(f"Sending email to {recipient_email}..."):
                        try:
                            dr.send_email(
                                recipient=recipient_email,
                                subject=subject,
                                body=edited_email,
                                docx_bytes=docx_bytes,
                                docx_filename=filename,
                            )
                            st.success(f"Email sent successfully to *{recipient_email}*.")
                        except ValueError as e:
                            st.error(str(e))
                            st.info(
                                "*Setup required:* Add these two lines to your .env file:\n\n"
                                "```\nSMTP_SENDER=your_gmail@gmail.com\n"
                                "SMTP_PASSWORD=xxxx xxxx xxxx xxxx\n```\n\n"
                                "The password must be a *Gmail App Password*, not your account password. "
                                "Generate one at: *Google Account → Security → 2-Step Verification → App Passwords*."
                            )
                        except Exception as e:
                            st.error(f"Failed to send email: {e}")