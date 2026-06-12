import streamlit as st
groq_api_key = st.secrets["GROQ_API_KEY"]

import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Executive Workforce Intelligence",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.main {
    background-color: #F8F9FB;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    color: #6B7280;
    font-size: 18px;
    margin-bottom: 30px;
}

.kpi-card {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}

.kpi-title {
    color: #6B7280;
    font-size: 14px;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #111827;
}

.insight-box {
    background-color: white;
    border-radius: 20px;
    padding: 30px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("📌 About Project")

    st.markdown("""
### Executive workforce insights 
 
### for organizational planning, talent visibility, and people strategy.

#### by S.Widjaja

---

#
""")


# =====================================================
# HEADER
# =====================================================
st.markdown(
    '<div class="big-title">📊 Executive Workforce Interactive</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Workforce analytics assistant for strategic HR insights</div>',
    unsafe_allow_html=True
)

# =====================================================
# KPI CARDS
# =====================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-title">Workforce Size</div>
        <div class="kpi-value">10</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-title">Business Units</div>
        <div class="kpi-value">5</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-title">Talent Coverage</div>
        <div class="kpi-value">100%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-title">Strategic Focus</div>
        <div class="kpi-value">People</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")
st.write("")


# =====================================================
# LLM
# =====================================================
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)


# =====================================================
# PROMPT
# =====================================================
PROMPT_TEMPLATE = """
You are an executive workforce analytics assistant.

Your role is to provide professional HR and workforce insights.

Only answer using the context.

<context>
{context}
</context>

Question:
{question}

Rules:
- Answer in same language
- Be executive and professional
- Use bullet points when useful
- Give strategic insights
- Use numbers/statistics when possible
- Do not hallucinate
- If unavailable say you don't know
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# =====================================================
# EMBEDDING
# =====================================================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# =====================================================
# PINECONE
# =====================================================
pc = Pinecone(api_key=pinecone_api_key)

index_name = "karyawan-data-384"

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =====================================================
# FORMAT DOCS
# =====================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# RAG CHAIN
# =====================================================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | groq_chat
    | StrOutputParser()
)


# =====================================================
# EXECUTIVE ASSISTANT
# =====================================================
st.markdown(
    '<div class="section-title">Executive Workforce Assistant</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Ask an executive workforce question",
    value=st.session_state.get("question", ""),
    height=120,
    placeholder="Example: Employee Overview ?"
)

if st.button("Enter"):

    if question.strip():

        with st.spinner("Analyzing workforce intelligence..."):

            response = rag_chain.invoke(question)

            st.write("")

            st.markdown(
                '<div class="section-title">Executive Insight</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    {response}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            with st.expander("📚 Retrieved Evidence (RAG Transparency)"):

                docs = retriever.invoke(question)

                for i, doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")
                    st.write(doc.page_content[:1000])

    else:
        st.warning("Please enter a question.")


st.divider()


# =====================================================
# EXECUTIVE QUESTION RECOMMENDATIONS
# =====================================================
st.markdown(
    '<div class="section-title">Executive Question Recommendations</div>',
    unsafe_allow_html=True
)

st.caption(
    "Choose a recommended workforce analysis question or write your own."
)


# =====================================================
# GROUP 1
# =====================================================
st.markdown("## 👥 Workforce Overview")

col1, col2 = st.columns(2)

with col1:

    if st.button("Executive Workforce Summary"):
        st.session_state.question = (
            "Provide an executive summary of workforce data."
        )

    if st.button("Employee Distribution Analysis"):
        st.session_state.question = (
            "How are employees distributed across divisions?"
        )

    if st.button("Workforce Demographic Observation"):
        st.session_state.question = (
            "What workforce demographic observations can management identify?"
        )


with col2:

    if st.button("Employee Data Quality"):
        st.session_state.question = (
            "Are there any employee data quality concerns?"
        )

    if st.button("Workforce Composition"):
        st.session_state.question = (
            "What does workforce composition look like?"
        )


st.divider()


# =====================================================
# GROUP 2
# =====================================================
st.markdown("## 💰 Compensation & Organization")

col3, col4 = st.columns(2)

with col3:

    if st.button("Compensation Structure Insight"):
        st.session_state.question = (
            "What compensation structure insights can management identify?"
        )

    if st.button("Salary Fairness Observation"):
        st.session_state.question = (
            "Are there any salary fairness or consistency observations?"
        )

    if st.button("Division Performance Potential"):
        st.session_state.question = (
            "Which divisions may need more management attention or support?"
        )


with col4:

    if st.button("Organizational Workforce Balance"):
        st.session_state.question = (
            "Does the workforce distribution appear balanced across divisions?"
        )

    if st.button("HR Efficiency Opportunity"):
        st.session_state.question = (
            "What HR efficiency opportunities may exist?"
        )


st.divider()


# =====================================================
# GROUP 3
# =====================================================
st.markdown("## 📈 Strategic Workforce Intelligence")

col5, col6 = st.columns(2)

with col5:

    if st.button("Strategic HR Insight"):
        st.session_state.question = (
            "What strategic HR insights can management identify?"
        )

    if st.button("Workforce Opportunity"):
        st.session_state.question = (
            "What workforce opportunities can management identify?"
        )

    if st.button("Leadership Insight"):
        st.session_state.question = (
            "What leadership insights can management learn from this workforce?"
        )


with col6:

    if st.button("Potential Workforce Risk"):
        st.session_state.question = (
            "Are there any workforce risks management should consider?"
        )

    if st.button("Future Workforce Planning"):
        st.session_state.question = (
            "What insights may help future workforce planning?"
        )
