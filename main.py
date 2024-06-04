from flask import Flask, request, abort, make_response , send_file
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3 import WebhookParser
from linebot.v3.messaging import TemplateMessage,ButtonsTemplate,URIAction,MessageAction,PostbackAction
from werkzeug.utils import secure_filename
import time
import dotenv
import os

dotenv.load_dotenv()

enableLLM = True
model = os.getenv('MODEL')
if(enableLLM):

    from langchain import hub
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    from langchain_community.document_loaders import TextLoader
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import ChatPromptTemplate

    # for LLM model
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
    
    from langchain_groq import ChatGroq
        
        
    loader = TextLoader("./index.md")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    print('完成向量化')
    

    # 進行檢索與生成
    retriever = vectorstore.as_retriever()
    
    #"""你是HorizonVision這個專題的製作者之一，你需要回答任何有關HorizonVision的問題。"""
    template = """你是HorizonVision這個專題的製作者之一，你需要以生動活潑的性格，回答任何有關HorizonVision的問題。 Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.使用台灣的繁體中文回覆。

Question: {question} 

Context: {context} 

Answer:"""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    # prompt = hub.pull("rlm/rag-prompt")
    print(prompt)
    
    
    if(model == 'chatgpt' ):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif(model == 'groq'):
        llm = ChatGroq(temperature=0.6, model_name="llama3-8b-8192")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("完成建立RAG")


app = Flask(__name__)

configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
parser = WebhookParser(os.getenv('LINE_CHANNEL_SECRET'))

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/file/<string:filename>", methods=['GET'])
def return_pdf(filename):
    try:
        filename = secure_filename(filename)  # Sanitize the filename
        file_path = os.path.join('files', filename)
        if os.path.isfile(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return make_response(f"File '{filename}' not found.", 404)
    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)

#import parser

template_message = TemplateMessage(
    alt_text='ButtonsTemplate',
    template=ButtonsTemplate(
        thumbnail_image_url='https://line.ljthub.com/file/HorizonVision.jpg',
        title='關於HorizonVision專題',
        text='常見問題',
        actions=[
            MessageAction(
                label='簡介',
                text='HorizonVision 簡介、特色及功能'
            ),
            MessageAction(
                label='作者及指導老師',
                text='HorizonVision 作者及指導老師是誰？'
            ),
            URIAction(
                label='操作影片',
                uri='https://youtu.be/dJg1ELYqn7Y'
            )
            
        ]
    )
)


@app.route("/horizonvision/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        # handler.handle(body, signature)
        # events = paser
        events =  parser.parse(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
        
        
    for event in events:
        with ApiClient(configuration) as api_client:
            print("="*20)
            #if timestamp is > than now - 1 minute, then skip
            if event.timestamp < time.time() - 10:
                print("skip")
                continue
            elif type(event) != MessageEvent:
                print("skip")
                continue
            print(event)
            
            line_bot_api = MessagingApi(api_client)
            if(enableLLM):
                print(f"Question: {event.message.text}")
                result = rag_chain.invoke(event.message.text)
                print(f"Answer: {result}")
                line_bot_api.reply_message_with_http_info( ReplyMessageRequest( reply_token=event.reply_token, messages=[TextMessage(text=result),template_message]))
            else:
                line_bot_api.reply_message_with_http_info( ReplyMessageRequest( reply_token=event.reply_token, messages=[TextMessage(text=event.message.text),template_message]))
                
            

    

    return 'OK'


if __name__ == "__main__":
    app.run(port=5001)
    