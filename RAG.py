from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout
from llama_index.core import  SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
import chromadb

# hyperparameters: CHUNK_SIZE, TOP_K
CHUNK_SIZE = 512
TOP_K = 2

# load LLM model quantified with MLC 4bit
llm = ChatModule(
    model="/data/RAG_based_on_Jetson/llama2-7b-MLC-q4f16-jetson-containers/params",
    model_lib_path="/data/RAG_based_on_Jetson/llama2-7b-MLC-q4f16-jetson-containers/Llama-2-7b-chat-hf-q4f16_ft-cuda.so"
    )

# build document
documents = SimpleDirectoryReader("/data/RAG_based_on_Jetson/data/").load_data()


# Use a Text Splitter to Split Documents
text_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    # separator=" ",
)
text_chunks = []

# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# Manually Construct Nodes from Text Chunks
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# Generate Embeddings for each Node
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-l")

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding


# create client and a new collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="my_collection")

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
vector_store.add(nodes)


while True:
    try:
        user_input = input('\033[94m' +"Prompt: " + '\033[0m')
        query_embedding = embed_model.get_query_embedding(user_input)
        query_mode = "default"
        vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=TOP_K, mode=query_mode)
        query_result = vector_store.query(vector_store_query)
        information = ''
        for i in range(TOP_K):
            information += query_result.nodes[0].get_content()
        prompt = f'You are a smart agent. A question would be asked to you and relevant information would be provided.\
    Your task is to answer the question and use the information provided. Question - {user_input}. Relevant Information - {information}'
        llm.generate(
            prompt=prompt,
            progress_callback=StreamToStdout(callback_interval=2),
            )
        print('**'*100)
        print(f'\033[1;34m externel imformation:\n {information}')
        
    except KeyboardInterrupt:
        break

