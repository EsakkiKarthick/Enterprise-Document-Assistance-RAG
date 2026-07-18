from langgraph.graph import StateGraph, END
from chunking import chunk_sentence_statment_aware
from RAG_Pipeline import vector_store
from typing import TypedDict
import requests

class RAGState(TypedDict):
    query: str
    retrieved_chunks: list[dict]
    context_window: str
    llm_response: str
    final_output: str
    sources: list[dict]

def query_validation(state: RAGState):
    query=state['query'].strip()
    if query and len(query)<1000:
        return {"query": query}
    else:
        print(f'User Query Should not be more than 1000 characters')
        #state['query']=''
        return {"query": query}
    
def router_function(state: RAGState) -> str:
    query=state.get('query','').strip()
    if query and len(query)<1000:
        return "retrieve_node"
    else:
        return "fail"
    
def retrieve_node(state: RAGState) -> RAGState:
    print('retrieve_data function started')
    RAGData=vs.search(state['query'])

    print(f'Retrieved Data Count:{len(RAGData)}')
    for i , data in enumerate(RAGData, 1):
        id=data.get("id","")
        text_content=data.get("text", "no data")
        score = data.get("score", 0.0)
        metadata=data.get("metadata", "")

        print(f"chunk{i}:")
        print(f"-> Id:{id}")
        print(f"-> Text:{text_content[:100]}...")
        print(f"-> Score:{score}")
        print(f"-> Metadata:{metadata}")

    return {"retrieved_chunks": RAGData}

def context_data(state: RAGState) -> RAGState:
    context=[]
    sources=[]
    chunks=state["retrieved_chunks"]
    if state.get('retrieved_chunks'):
        for result in chunks:
            metadata=result.get('metadata', {})
            source=metadata.get('source','')
            page=metadata.get('page', '')
            extracted_date=metadata.get('extracted_date', '')
            context.append(f'source:{source}|Page No:{page}, [text]:{result["text"]}, Score: {result["score"]}')

            sources.append({
                "id" : {result['id']},
                "text":{result['text']},
                "score":{result['score']},
                "source":{source},
                "page":{page},
                "extracted_date":{extracted_date}
            })
            contex_window="\n".join(context)
        #print(f'Context Data:{contex_window}, Sources:{sources}')
        print(f'Context Data: {len(contex_window)} chars, {len(sources)} sources')
        #state['context_window']=contex_window
        #state['sources']=sources
        return {"context_window":contex_window, "sources": sources}
    else:
        print(f'No retrieved_chunks returned from RAG search')
        return {"context_window":"", "sources": []}

def generate_data(state: RAGState)-> RAGState:
    timeout_seconds=300
    query = state['query']
    context = state['context_window']
    systemPrompt=("You are a helpfull assistance agent. You will provide answer based on the context provided."
                    "Answer only based on context, if data not exists in context say 'I don't know the answer'")
    if context:
        prompt = f'{systemPrompt}\n\n{context}\n\n Question:{query}\n\n Answer:'
    else:
        prompt=f'{systemPrompt}\n\n Question:{query}\n\n Answer:'
    #print(f'LLM Call. Prompt{prompt}')
    try:
        response = requests.post("http://localhost:11434/api/generate",
                                 json={
                                     "model" : "llama3",
                                     "prompt" : prompt,
                                     "top_p" : 0.9,
                                     "stream": False
                                 }, timeout=timeout_seconds
                                )
        result=response.json()
        llm_response=result.get("response", "").strip()
        #print(f'LLM Response: {llm_response}')
        return {"llm_response":llm_response}

    except requests.exceptions.ConnectionError:
        print('Ollama service not available')
        return {"llm_response":'Ollama service not available'}
    except Exception as e:
        print(f'Exception Occured. Exception:{str(e)}')
        return {"llm_response":f'Exception Occured. Exception:{str(e)}'}

def format_output(state: RAGState) -> RAGState:
    response=state['llm_response']
    sources=state['sources']
    output_data=[response]
    if sources:
        output_data.append("-------------Sources------------")
        for i, source in enumerate(sources,1):
            output_data.append(
                f'{i}. Source: {source['source']}|Page: {source["page"]}|Score: {source["score"]}|'
                f'extracted_date: {source["extracted_date"]}'
            )
            final_output="\n".join(output_data)
        print(f'Final Output: {final_output}')


    return {"final_output":final_output}

if __name__=="__main__":
    COLLECTION_NAME="Employee-Handbook"
    vs=vector_store(collection_name=COLLECTION_NAME)
    graph=StateGraph(RAGState)

    graph.add_node("validate_node", query_validation)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("synthesize_context_node", context_data )
    graph.add_node("generate_node", generate_data)
    graph.add_node("format_output_node", format_output)

    graph.set_entry_point("validate_node")

    graph.add_conditional_edges("validate_node", router_function,
                                {
                                    "retrieve_node": "retrieve_node",
                                    "fail": END

                                })
    graph.add_edge("retrieve_node", "synthesize_context_node")
    graph.add_edge("synthesize_context_node", "generate_node")
    graph.add_edge("generate_node", "format_output_node")

    app=graph.compile()
    result=app.invoke({"query":"Whom should i reachout for Harassment?"})
    #print(f'Final Result:{result}')




