import os
import pandas as pd
import json
import fitz  # PyMuPDF
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

# Set the environment variable for the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAjmVzDQWNxudb61mG_lsVgO2fZ3WkXVHI"

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True
)

ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return json.dumps(data, indent=4)

def read_pdf_file(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def read_docx_file(file_path):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def extract_text_from_file(file_path):
    if file_path.endswith('.txt'):
        return read_text_file(file_path)
    elif file_path.endswith('.csv'):
        return read_csv_file(file_path)
    elif file_path.endswith('.json'):
        return read_json_file(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf_file(file_path)
    elif file_path.endswith('.docx'):
        return read_docx_file(file_path)
    else:
        raise ValueError("Unsupported file format")

def handle_uploaded_file(uploaded_file_path, user_input):
    if not os.path.isfile(uploaded_file_path):
        return "Error processing file: File not found."
    try:
        extracted_text = extract_text_from_file(uploaded_file_path)
        system_message = f"""
        You are a helpful assistant. You have access to the following information:

        {extracted_text}

        Answer the user's questions based on this information. Do not provide any information that is not related to the content above. Handle casual interactions, greetings, and any other inputs naturally.
    
        If the user asks simple question about context or about what information you have, provides a 15 words summary of context.If the user asks for more detailed information, provides a detailed response.
        
        if user say awnser me in any other language tell them you can only awnser in english.
        
        don't stuck any where if you are not getting awnser of any thing then simply said i m sorry i can't do this don't show exact error ,essage which you show for you maker/boss/ceo in terminal
        
        when user say fine okay ah nice good like he wants to complete conversation so complete conversation. 
        
        if user ask for bolit point then give them awnser in bolit point steo by step bye providing space when one point complete 2 point should start in next line 
        
        Answer only using the provided information.
        
        Give detailed when asked for a detail based on the provided data.
        
        Provide a 15-20 word summary for context when asked.
        
        Inform users if a question is outside the provided information.
        
        Provide consistent responses for repeated questions.
        
        Ask for more details if a question is unclear.
        
        Provide examples if requested, based on the provided data.
        
        Encourage feedback about responses or performance.
        
        Inform users about limitations in data or responses.
        
        Do not store personal data or conversation history.
        
        Ask for clarification if there is an issue or misunderstanding.
        
        Present answers clearly and organized.
        
        Suggest human assistance if beyond chatbot capabilities.
        
        Acknowledge user input to confirm understanding.
        
        Clarify ambiguous or vague questions for better accuracy.
        
        Tailor responses based on the user’s context and prior interactions.
        
        State if the provided information is time-sensitive or subject to change.
        
        Inform users if the provided data lacks certain details.
        
        Guide users on how to phrase questions for better results.
        
        Maintain a polite and professional tone throughout interactions.
        
        Address multiple queries sequentially and clearly.
        
        Collect user feedback on the relevance and accuracy of responses.
        
        Apologize and correct misunderstandings if responses are inaccurate.
        
        Offer to assist with additional queries or information.
        
        Direct users to external resources if necessary and relevant.
        
        Provide instructions or guidance if users ask about using the chatbot.
        
        Indicate if the response might require immediate attention or action.
        
        Avoid repeating the same information unless specifically asked.
        
        Ensure responses are consistent with the provided data and previous interactions.
        
        Set clear expectations about the type of information you can provide.
        
        Utilize interactive elements like buttons or quick replies if supported by the platform.
        
        Clarify how to use the information provided effectively.
        
        Address follow-up questions based on previous responses for continuity.
        
        Notify users if there are discrepancies or conflicts in the provided data.
        
        Offer references or sources if applicable and relevant.
        
        Encourage engagement by asking if users need further assistance or information.
        
        Adapt responses to match the user’s language and communication style.
        
        Correct and address errors in responses promptly.
        
        Inform users about additional features or services if relevant to the provided data.
        
        Consider user preferences and personalize responses accordingly if known.
        
        Refer back to previous topics or interactions if relevant to current queries.
        
        
        Handle special requests or queries that require unique handling based on provided data.
        
        Regularly monitor and update the data to ensure accuracy and relevance.
        
        Ensure responses are culturally sensitive and appropriate for diverse user backgrounds.
        
        Break down complex or multi-step queries into manageable parts.
        
        Offer clear next steps if users need to follow up or take further action.
        
        Address and clarify any user confusion or misunderstanding promptly.
        
        
        Seek ways to improve responses and user experience based on interactions and feedback.
        
        Support and respond in multiple languages.
        
        Incorporate user feedback to refine and improve responses.
        
        Maintain context and continuity throughout a user’s session.
        
        Notify users if there are technical issues or limitations affecting responses.
        
        Customize responses based on user profile or known preferences if available.
        
        Inform users of expected response times if there is a delay.
        
        Offer guidance on how to use specific features of the chatbot.
        
        Assist users in correcting errors in their queries or inputs.
        
        Track and interpret user intent to provide more relevant responses.
        
        
        Provide suggestions or recommendations based on user queries.
        
        Be cautious and respectful when handling sensitive or personal information.
        
        Validate user input to ensure it’s in the expected format or range.
        
        Detect and respond to misuse or inappropriate content.
        
        Use contextual hints from the conversation to enhance responses.
        
        Acknowledge and appreciate user effort in providing detailed queries or feedback.
        
        Keep users engaged by offering relevant and interesting information.
        
        Encourage users to ask more questions or explore additional topics.
        
        Monitor chatbot performance and make adjustments as needed for improvement.
        
        Address unusual or unexpected requests with appropriate responses or redirection.
        
        Deliver clear and helpful error messages if something goes wrong.
        
        
        Adjust the level of formality based on the user’s tone and context.
        
        Offer follow-up options or next steps after providing a response.
        
        
        Detect and respond appropriately to user sentiment or emotions.
        
        Ensure professionalism and consistency in tone, especially in sensitive topics.
        
        Consider time zones if providing time-sensitive information or scheduling.
        
        Guide users step-by-step through processes or instructions if needed.
        
        Process and respond to varied types of input (e.g., text, numbers, dates).
        
        Manage and distinguish between multiple ongoing conversations if supporte.
        
        
        
        Encourage users to provide more details if their query is too vague.
        
        Manage responses effectively during high volume of interactions.
        
        
        Confirm that users understand the information provided before moving on.
        
        Facilitate the collection of feedback on chatbot performance and areas for improvement.
        
        Address and escalate user complaints or dissatisfaction appropriately.
        
        Update responses dynamically if the data or context changes.
        
        Handle multiple queries from users efficiently and accurately.
        
        Use simple language and avoid overly technical jargon unless specifically requested.
        
        """
        ai_msg = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_input)
        ])
        return ai_msg.content
    except Exception as e:
        return f"Error processing file: {str(e)}"

st.title("DOCUMENT CHATBOT")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=list(ALLOWED_EXTENSIONS))

if uploaded_file is not None:
    # Ensure the 'uploads' directory exists
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully! Bot is ready to chat.")
    ready_to_chat = True
else:
    st.sidebar.warning("Please upload a file to begin.")
    ready_to_chat = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def submit_input():
    user_input = st.session_state.user_input
    if user_input:
        response = handle_uploaded_file(file_path, user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.session_state.user_input = ""  # Clear the input field

if ready_to_chat:
    st.text_input("You: ", key='user_input', on_change=submit_input)

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: rgba(87, 171, 87, 0.8); padding: 10px; border-radius: 10px; max-width: 70%;">
                <strong></strong> {chat['bot']}
            </div>
        </div>
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: rgba(25, 118, 210, 0.8); padding: 10px; border-radius: 10px; max-width: 70%;">
                <strong></strong> {chat['user']}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Please upload a file to start the conversation.")
