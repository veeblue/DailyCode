import streamlit as st

from rag_module import answer_question  # 引入你前面写的函数

st.title("📚 RAG 问答系统")
question = st.text_input("请输入你的问题：")

if question:
    with st.spinner("正在思考中..."):
        
        answer, sources = answer_question(question)

        st.success(answer)

        st.markdown("---")
        st.markdown("#### 🔍 参考文段：")
        for i, doc in enumerate(sources):
            st.markdown(f"**文段 {i+1}：**\n{doc.page_content[:300]}…")