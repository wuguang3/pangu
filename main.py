import os
import time
import tkinter
import traceback
from threading import Thread

from langchain import hub
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


class PanGu:
    retry_times = 3
    model = "gpt-4"
    base_url=""
    api_key=os.environ.get("OPENAI_API_KEY", "")
    serpapi_api_key = os.environ.get("SERPAPI_API_KEY", "")
    persist_directory = "./knowledge_store"
    base_directory = os.path.dirname(os.path.dirname(__file__))

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model=self.model, base_url=self.base_url, api_key=self.api_key)
        self.agent = None
        self.prompt = None
        self.tools = []

        self.init_prompt()
        self.init_tools()
        self.init_agent()
        print(self.base_directory)

    def init_prompt(self):
        self.prompt = ChatPromptTemplate.from_template("""
            SYSTEM
            你是一个非常有帮助的AI助手,你会简短地运用成语回答问题。
            PLACEHOLDER
            {{chat_history}}
            HUMAN
            {input}
            PLACEHOLDER
            {agent_scratchpad}
        """)

    def init_tools(self):
        os.makedirs(os.path.join(self.base_directory, 'knowledge'), exist_ok=True)
        documents = DirectoryLoader(os.path.join(self.base_directory, 'knowledge'), glob="**/*.txt").load()
        embeddings = OpenAIEmbeddings(openai_api_base=self.base_url, openai_api_key=self.api_key)
        if documents:
            try:
                text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
                docs = text_splitter.split_documents(documents)
                vector = Chroma.from_documents(docs, embeddings)

                self.tools.append(
                    create_retriever_tool(
                        vector.as_retriever(),
                        "common_search",
                        description="useful for when you need to answer questions, from local knowledge base, Regardless of whether it is right or wrong, it is returned directly to the user."
                    )
                )
            except Exception as e:
                traceback.print_exc()
                pass
        try:
            docsearch = Chroma(persist_directory=os.path.join(self.base_directory, 'knowledge_store'), embedding_function=embeddings)

            self.tools.append(
                create_retriever_tool(
                    docsearch.as_retriever(),
                    "Search",
                    description="useful for when you need to answer questions about 吴广 虞树燕， 你会很简短地回答问题并且善于用成语回答"
                )
            )
        except Exception as e:
            traceback.print_exc()
            pass

        self.tools.append(
            Tool(name='google_search', description="Search web for recent results.",
                 func=SerpAPIWrapper(params={
                        "engine": "google",
                        "google_domain": "google.com",
                        "gl": "us",
                        "hl": "en"}, serpapi_api_key=self.serpapi_api_key
                     ).run
            )
        )

    def init_agent(self):
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)

    def ask(self, text: str, ret: list) -> str:
        try:
            result = AgentExecutor(agent=self.agent, tools=self.tools).invoke({"input": text})
        except TimeoutError:
            print("超时")
            return "超时"
        else:
            print(result['output'])
            ret.append(result['output'])
            return result


class PGUI:
    def __init__(self):
        self.pg = PanGu()
        self.root = tkinter.Tk()
        self.root.title("少侠请提问")
        self.center_window(550, 220)
        self.root.resizable(False, False)

    def center_window(self, w, h):
        # 获取屏幕 宽、高
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        # 计算 x, y 位置
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def btn_handler(self, input_box, text: tkinter.Text):
        question = input_box.get()

        result = []
        task = Thread(target=self.pg.ask, args=(question, result))
        task.start()

        times, total = 1, 0
        while not result:
            if total >= 300:
                result.append("这题俺不会")
            if times > 6:
                times = 1
            text.delete("1.0", tkinter.END)
            text.insert(tkinter.INSERT, "思考中" + "." * times)
            self.root.update()
            time.sleep(1)
            times += 1
            total += 1

        text.delete("1.0", tkinter.END)
        text.insert(tkinter.INSERT, result[0])
        self.root.update()

    def input(self):
        input_box = tkinter.Entry(self.root, width=60, textvariable=tkinter.StringVar(value="唐朝有几位皇帝"))
        input_box.grid(row=0, column=0, padx=10, pady=10)
        return input_box

    def button(self, input_box, text):
        btn = tkinter.Button(self.root, text="确定", width=10, height=1, command=lambda: self.btn_handler(input_box=input_box, text=text))
        btn.grid(row=0, column=1, padx=10, pady=10)
        return btn

    def result_text(self):
        label = tkinter.Text(self.root, width=75, height=10)
        label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        return label

    def show(self, *args, **kwargs):
        input_box = self.input()
        text = self.result_text()
        btn = self.button(input_box, text)
        self.root.mainloop()


if __name__ == '__main__':
    pgui = PGUI()
    pgui.show()