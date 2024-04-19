# RAG_based_on_Jetson
This project has implemented the RAG function on Jetson and supports TXT and PDF document formats. It uses MLC for 4-bit quantization of the Llama2-7b model, utilizes ChromaDB as the vector database, and connects these features with Llama_Index. I hope you like this project.

# Hardware Prepare
For this project you need Jetson Orin NX 16GB, this project will use RAM at a peak of 11.7GB. For myself, I use [Recomputer J4012](https://www.seeedstudio.com/reComputer-J4012-w-o-power-adapter-p-5628.html).

# Run this project
## Step 1: prepare environment

```git clone --depth=1 https://github.com/dusty-nv/jetson-containers```

```cd jetson-containers pip install -r requirements.txt && cd data```

```git clone https://github.com/Seeed-Projects/RAG_based_on_Jetson.git```

```cd RAG_based_on_Jetson && git clone https://huggingface.co/JiahaoLi/llama2-7b-MLC-q4f16-jetson-containers && cd ..```

## Step 2: run and enter the docker 

```cd .. && ./run.sh $(./autotag mlc) ```

```cd data/RAG_based_on_Jetson && pip install -r requirements.txt```

``` pip install chromadb==0.3.29```

## step 3: run the project

```python3 main.py```

# Result 
![./simRAG.mkv]
