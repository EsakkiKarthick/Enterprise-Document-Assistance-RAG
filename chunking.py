import tiktoken
import re
def chunk_text(text: str, max_tokens: int=250, chunk_overlap: int=50, model: str='gpt-4' ) -> list [str]:
	
    enc=tiktoken.encoding_for_model(model)
    tokens=enc.encode(text)
    chunk_data=[]
    for i in range(0, len(tokens), max_tokens-chunk_overlap):
        chunk_token=tokens[i:i+max_tokens]
        chunk_data.append(chunk_token)
    
    return chunk_data

def chunk_sentence_statment_aware(text: str, max_tokens: int=50, sentence_overlap: int=1):
    sentences=re.split(r'(?<=[.!?])\s+', text.strip())
    current_chunk=[]
    chunks=[]
    current_length=0
    for sentence in sentences:
        token_count=len(sentence.split())
        if current_length +token_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk=current_chunk[-sentence_overlap:]
            current_length=sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length+=token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk)) 
    #print('Chunk Data', chunks)

    return chunks

if __name__=="__main__":
    input_data= """
                Employees are entitled to 20 days of paid leave annually. Sick leave can be taken with prior approval. In case of emergency, employees must inform their manager within 24 hours.

                Unused leave can be carried forward up to a maximum of 10 days. Leave beyond this limit will lapse at the end of the year.
                """
    chunks=chunk_sentence_statment_aware(input_data)
    print(f'Chunks Count:{len(chunks)}')
    for i , c in enumerate(chunks):
        token_data=chunk_text(c)
        print(f'Chunk {i+1}: {c}')
        print(f'Token Data: {token_data}')
