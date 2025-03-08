import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Interface Streamlit
st.title('Chatbot PDF avec ChatGPT')

# Télécharger le PDF
pdf_file = st.file_uploader('Téléchargez votre PDF', type='pdf')

if pdf_file is not None:
    # Extraire le texte
    text = extract_text_from_pdf(pdf_file)
    
    # Diviser le texte en morceaux
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Créer des embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # Initialiser le chatbot
    llm = OpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever())
    
    # Interface de chat
    st.write('Posez vos questions sur le PDF :')
    user_question = st.text_input('Votre question')
    if user_question:
        response = qa.run(user_question)
        st.write(response)
