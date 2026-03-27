



import json
from typing import List, Tuple, Dict, Set
import time
import pandas as pd


from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import re
import argparse
import ast


def load_model():
    model_id = "google/gemma-3-27b-it-qat-q4_0-gguf"
    model_basename = "gemma-3-27b-it-q4_0.gguf" 

    
    model_path = hf_hub_download(
        repo_id=model_id,
        filename=model_basename,
        resume_download=True
    )

    print(f"Model downloaded to: {model_path}")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=8192,
        n_batch=512,
        
    )
    return llm



def llm_enrich_freetext(answer: str, llm: Llama) -> List[Dict[str, object]]:

    promptRaw = f"""
Esegui l'analisi del sentiment e delle emozioni per il TESTO racchiuso tra i tag [TESTO] e [/TESTO].

ISTRUZIONI OBBLIGATORIE (seguile alla lettera):
1. Il sentiment deve essere **esattamente uno** dei seguenti tre valori:
   ["positivo", "neutro", "negativo"]  
   Non usare sinonimi, abbreviazioni o altre parole.

2. Le emozioni devono essere **esclusivamente scelte** tra le seguenti otto emozioni di base di Plutchik:
   ["gioia", "tristezza", "fiducia", "disgusto", "anticipazione", "sorpresa", "rabbia", "paura"]  
   Non inventare nuove emozioni e non modificare i nomi (es. niente "entusiasmo", "delusione", ecc.).  
   Se nessuna emozione è rilevante, restituisci una lista vuota: [].

3. Se il testo esprime emozioni simili a quelle elencate, **mappale alla più vicina** tra le 8 permesse, senza crearne di nuove.

4. L'output deve essere **solo e unicamente** un dizionario JSON conforme al seguente formato.  
   Non aggiungere spiegazioni, testo extra o commenti prima o dopo.

Formato richiesto (rispetta virgolette e struttura esattamente come segue):

{ 
  "TESTO": [
    sentiment,
    [emozione1, emozione2]
  ]
} 

Ecco il testo da analizzare:
[TESTO]{answer}[/TESTO]

Prima di restituire l'output, verifica che sia un JSON valido e conforme alle istruzioni sopra.
"""

    
    prompt = f"""<start_of_turn>user
    {promptRaw}

    <end_of_turn>
    <start_of_turn>model
    """

    TEMPERATURE=0.0

    output = llm(
        prompt,
        max_tokens=5000,
        temperature=TEMPERATURE,
        top_p=0.95,
        stop=["###"],
        echo=False,
        stream=False
    )

    enriched = output["choices"][0]["text"].replace("```json", "").replace("```","").strip()
    
    try:
        
        
        enriched = json.loads(enriched)
        _,value = enriched.popitem() 
        enriched[answer] = value
        print(f"\nTEXT: '{answer}':\nOUTPUT:{enriched[answer]}\n")
        
        return enriched
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON string: {e}")
        return {}


def llm_enrich_word(word: str, llm: Llama) -> List[Dict[str, object]]:

    promptRaw = f"""
Esegui l'analisi del sentiment e delle emozioni per il TESTO racchiuso tra i tag [TESTO] e [/TESTO].

ISTRUZIONI OBBLIGATORIE (seguile alla lettera):
1. Il sentiment deve essere **esattamente uno** dei seguenti tre valori:
   ["positivo", "neutro", "negativo"]  
   Non usare sinonimi, abbreviazioni o altre parole.

2. Le emozioni devono essere **esclusivamente scelte** tra le seguenti otto emozioni di base di Plutchik:
   ["gioia", "tristezza", "fiducia", "disgusto", "anticipazione", "sorpresa", "rabbia", "paura"]  
   Non inventare nuove emozioni e non modificare i nomi (es. niente "entusiasmo", "delusione", ecc.).  
   Se nessuna emozione è rilevante, restituisci una lista vuota: [].

3. Se il testo esprime emozioni simili a quelle elencate, **mappale alla più vicina** tra le 8 permesse, senza crearne di nuove.

4. L'output deve essere **solo e unicamente** un dizionario JSON conforme al seguente formato.  
   Non aggiungere spiegazioni, testo extra o commenti prima o dopo.

Formato richiesto (rispetta virgolette e struttura esattamente come segue):

{ 
  "TESTO": [
    sentiment,
    [emozione1, emozione2]
  ]
} 

Ecco il testo da analizzare:
[TESTO]{word}[/TESTO]

Prima di restituire l'output, verifica che sia un JSON valido e conforme alle istruzioni sopra.
"""
    
    
    prompt = f"""<start_of_turn>user
    {promptRaw}

    <end_of_turn>
    <start_of_turn>model
    """

    TEMPERATURE=0.0

    output = llm(
        prompt,
        max_tokens=5000,
        temperature=TEMPERATURE,
        top_p=0.95,
        stop=["###"],
        echo=False,
        stream=False
    )

    enriched = output["choices"][0]["text"].replace("```json", "").replace("```","").strip()
    
    try:
        print(f"Output for word '{word}': {enriched}")
        enriched = json.loads(enriched)
        _,value = enriched.popitem() 
        enriched[word] = value
        print(f"\nWORD: '{word}':\nOUTPUT:{enriched[word]}\n")
        
        return enriched
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON string: {e}")
        return {}






def checkOutput(enriched):
    
    valid_sentiments = {"positivo", "neutro", "negativo"}
    valid_emotions = {"gioia", "tristezza", "fiducia", "disgusto", "anticipazione", "sorpresa", "rabbia", "paura"}
    
    
    invalid_sentiment_found = False

    
    for item_key, item_value in enriched.items():
        
        if not isinstance(item_value, (list, tuple)) or len(item_value) != 2:
            print(f"Structural Error: Value for key '{item_key}' is not a list/tuple of two elements: {item_value}")
            return False

        sentiment = item_value[0]
        current_emotions = item_value[1]
        
        
        if sentiment not in valid_sentiments:
            print(f"Validation Error: Invalid sentiment '{sentiment}' for key '{item_key}'. Expected one of {valid_sentiments}.")
            invalid_sentiment_found = True
            
            
            
        
        if isinstance(current_emotions, (list, set, tuple)):
            
            filtered_emotions = [
                emotion for emotion in current_emotions 
                if emotion in valid_emotions
            ]
            
            
            
            enriched[item_key] = [sentiment, filtered_emotions]
            
            
            if len(filtered_emotions) < len(current_emotions):
                 invalid_removed = set(current_emotions) - set(filtered_emotions)
                 print(f"Emotion Filter: Removed invalid emotions {invalid_removed} from key '{item_key}'.")
        else:
             print(f"Structural Error: Emotion value for key '{item_key}' is not a list/set/tuple: {current_emotions}")
             return False 

    
    if invalid_sentiment_found:
        print("\nChecks finished. Invalid sentiments were found, returning False.")
        return False
    else:
        print("\nAll checks passed and emotions were filtered: 'enriched' dictionary is valid.")
        return True


def _sanitize_json_key(text: str) -> str:
    if pd.isna(text) or text is None:
        return text
    
    
    text = re.sub(r'[\n\t\r]', ' ', text)
    
    
    
    
    
    
    text = re.sub(r'\s+', ' ', text)
    
    
    return text.strip()


def extract_unique_strings_fromText(input_txt_filepath: str) -> List[str]:
    
    try:
        
        with open(input_txt_filepath, mode='r', encoding='utf-8') as f:
            file_content = f.read().strip()
            
        
        
        unique_set: Set[str] = ast.literal_eval(file_content)
        
        
        if not isinstance(unique_set, set):
            print(f"Error: File content was evaluated, but the result is not a set. Type found: {type(unique_set)}")
            return []
            
        
        
        filtered_strings = [_sanitize_json_key(s) for s in unique_set if isinstance(s, str) and len(s) >= 10]

        return filtered_strings
        
    except FileNotFoundError:
        print(f"Error: File not found at path: {input_txt_filepath}")
        return []
    except ValueError as e:
        print(f"Error: Failed to evaluate file content as a set literal. Check file formatting. Details: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []




def extract_unique_strings_freetext(input_excel: str, columns: List[str]) -> List[str]:
    try:
        df = load_simplified_dataframe(input_excel)
    except ValueError as e:
        print(f"Error loading Excel data: {e}")
        return []
    
    unique_strings = set()
    
    for col in columns:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            filtered_values = values[values.str.len() >= 3]
            unique_strings.update(filtered_values.tolist())
        else:
            print(f"Warning: Column '{col}' not found in the simplified header.")
    
    return list(unique_strings)


def extract_unique_strings_word(input_excel: str, columns: List[str]) -> List[str]:
    try:
        df = load_simplified_dataframe(input_excel)
    except ValueError as e:
        print(f"Error loading Excel data: {e}")
        return []
    
    unique_strings = set()
    
    for col in columns:
        if col in df.columns:
            
            values = df[col].dropna().astype(str).str.strip()
            filtered_values = values[values.str.len() >= 3]
            unique_strings.update(filtered_values.tolist())
        else:
            print(f"Warning: Column '{col}' not found in the simplified header.")
    
    return list(unique_strings)




def computeSentimentEmotions_freetext(strings: List[str], llm: Llama) -> Dict[str, Tuple[str, List[str]]]:
    enriched_dict = {}
    to_be_processed: Set[str] = set()
    retry_counts: Dict[str, int] = {}  
    MAX_RETRIES = 10
    
    processedSoFar = 0
    for testo in strings:
        
        callLLMandCheck_freetext(llm, enriched_dict, to_be_processed, testo)
        if testo in to_be_processed:
            retry_counts[testo] = retry_counts.get(testo, 0) + 1
            
        processedSoFar += 1
        print(f"Processed {processedSoFar} items so far of {len(strings)} total items. Check: {len(enriched_dict)}")
        print(f"Currently to be processed: {len(to_be_processed)} items: {to_be_processed}")            
        
    
    if to_be_processed:
        print("\nAttempting to process remaining items with retries...")
        
        
        while to_be_processed:
            items_to_retry = list(to_be_processed) 
            
            
            
            items_failed_this_pass = set() 
            
            print(f"--- Starting new retry pass with {len(items_to_retry)} items. ---")
            
            for failed_item in items_to_retry:
                
                if retry_counts[failed_item] < MAX_RETRIES:
                    
                    
                    to_be_processed.discard(failed_item) 
                    
                    callLLMandCheck_freetext(llm, enriched_dict, items_failed_this_pass, failed_item)
                    
                    if failed_item in items_failed_this_pass:
                        
                        to_be_processed.add(failed_item) 
                        retry_counts[failed_item] += 1
                        
            
            newly_failed_and_eligible = {item for item in to_be_processed if retry_counts[item] < MAX_RETRIES}
            
            if not newly_failed_and_eligible:
                break 
            
            print(f"After this pass: {len(to_be_processed)} items remaining. Max retries hit for {len(to_be_processed) - len(newly_failed_and_eligible)} items.")
            
    
    
    
    final_failed_to_process = {item for item in to_be_processed if retry_counts[item] >= MAX_RETRIES}
    
    print("\n--- Final Status ---")
    print(f"Successfully processed: {len(enriched_dict)} items.")
    print(f"Failed to process (Max retries hit): {len(final_failed_to_process)} items.")
    print(f"Failed items: {final_failed_to_process}")
    
    return enriched_dict




def computeSentimentEmotions_word(strings: List[str], llm: Llama) -> Dict[str, Tuple[str, List[str]]]:
    enriched_dict = {}
    to_be_processed: Set[str] = set()
    retry_counts: Dict[str, int] = {}  
    MAX_RETRIES = 10
    
    processedSoFar = 0
    for testo in strings:
        
        callLLMandCheck_word(llm, enriched_dict, to_be_processed, testo)
        if testo in to_be_processed:
            retry_counts[testo] = retry_counts.get(testo, 0) + 1
            
        processedSoFar += 1
        print(f"Processed {processedSoFar} items so far of {len(strings)} total items. Check: {len(enriched_dict)}")
        print(f"Currently to be processed: {len(to_be_processed)} items: {to_be_processed}")            
        
    
    if to_be_processed:
        print("\nAttempting to process remaining items with retries...")
        
        
        while to_be_processed:
            items_to_retry = list(to_be_processed) 
            
            
            
            items_failed_this_pass = set() 
            
            print(f"--- Starting new retry pass with {len(items_to_retry)} items. ---")
            
            for failed_item in items_to_retry:
                
                if retry_counts[failed_item] < MAX_RETRIES:
                    
                    
                    to_be_processed.discard(failed_item) 
                    
                    callLLMandCheck_word(llm, enriched_dict, items_failed_this_pass, failed_item)
                    
                    if failed_item in items_failed_this_pass:
                        
                        to_be_processed.add(failed_item) 
                        retry_counts[failed_item] += 1
                        
            
            newly_failed_and_eligible = {item for item in to_be_processed if retry_counts[item] < MAX_RETRIES}
            
            if not newly_failed_and_eligible:
                break 
            
            print(f"After this pass: {len(to_be_processed)} items remaining. Max retries hit for {len(to_be_processed) - len(newly_failed_and_eligible)} items.")
            
    
    
    
    final_failed_to_process = {item for item in to_be_processed if retry_counts[item] >= MAX_RETRIES}
    
    print("\n--- Final Status ---")
    print(f"Successfully processed: {len(enriched_dict)} items.")
    print(f"Failed to process (Max retries hit): {len(final_failed_to_process)} items.")
    print(f"Failed items: {final_failed_to_process}")
    
    return enriched_dict





def callLLMandCheck_freetext(llm, enriched_dict, to_be_processed, testo):
    enriched = llm_enrich_freetext(testo, llm)
    if enriched and checkOutput(enriched):
        enriched_dict.update(enriched)
    else:
        to_be_processed.add(testo)

def callLLMandCheck_word(llm, enriched_dict, to_be_processed, testo):
    enriched = llm_enrich_word(testo, llm)
    if enriched and checkOutput(enriched):
        enriched_dict.update(enriched)
    else:
        to_be_processed.add(testo)



def save_enriched_data(data: Dict[str, Tuple[str, List[str]]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_simplified_dataframe(input_excel: str) -> pd.DataFrame:
    
    df = pd.read_excel(input_excel, dtype=str, header=0)
    
    return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute sentiment and emotions for words using LLM. Needs two arguments, base_folder and base_file_name.")
    
    
    parser.add_argument(
        "base_folder", 
        nargs="?",  
        default="/mnt/sharedHD/CodeRepo/erickson/data/", 
        type=str, 
        help="this is the base folder where the input file is located."
    )
    
    
    parser.add_argument(
        "base_file_name", 
        nargs="?", 
        default="Q 2025_12 settembre 2025_15.23.xlsx", 
        type=str, 
        help="this is the localname of the input file (without path) containing the excel data."
    )
    
    args = parser.parse_args()

    
    base_folder = args.base_folder
    base_file_name = args.base_file_name

    
    INPUT_EXCEL = base_folder+base_file_name
    COLUMNS_OF_INTEREST = [ "Positiva",
                            "Negativa"]  
    
    llm = load_model()

    print(f"Using input excel file: {base_file_name} in folder: {base_folder}")
    print(f"processing columns: {COLUMNS_OF_INTEREST}")

    
    for col in COLUMNS_OF_INTEREST:
        unique_strings = extract_unique_strings_freetext(INPUT_EXCEL, [col])
        
        start_llm_process_time = time.perf_counter()
        enriched_data = computeSentimentEmotions_freetext(unique_strings, llm)
        end_llm_process_time = time.perf_counter()
        elapsed_llm_process_time = end_llm_process_time - start_llm_process_time
        print(f"LLM process + Validation execution time: {elapsed_llm_process_time:.6f} seconds") 
        ENRICHED_JSON = INPUT_EXCEL+f'_enriched_data.{col}.json'   
        save_enriched_data(enriched_data, ENRICHED_JSON)

    print(f"\nProcessing completed for {COLUMNS_OF_INTEREST}")

    
    
    
    
    
    
    
    
    
    
    
    
    

    COLUMNS_OF_INTEREST = [ "Sentiment parole_1: - Aula di sostegno",
                            "Sentiment parole_2: - PEI",
                            "Sentiment parole_3: - Insegnante di sostegno",
                            "Sentiment parole_4: - Co-docenza/collaborazione curricolare-sostegno",
                            "Sentiment parole_5: - Gruppo classe",
                            "Sentiment parole_6: - Personalizzazione didattica",
                            "Sentiment parole_7: - Educatori/assistenti autonomia e comunicazione",
                            "Sentiment parole_8: - Famiglia dell'alunno/a con disabilità",
                            "Sentiment parole_9: - Progetto di Vita",
                            "Sentiment parole_10: - Collaborazione coi servizi sanitari",
                            "Sentiment parole_11: - Autodeterminazione",
                            "Sentiment parole_12: - GLO",
                            "Sentiment parole_13: - Valutazione (dell’alunno/a con disabilità)",
                            "Sentiment parole_14: - Dirigente scolastico/a"] 

    unique_strings = extract_unique_strings_word(INPUT_EXCEL, COLUMNS_OF_INTEREST)
    
    start_llm_process_time = time.perf_counter()
    enriched_data = computeSentimentEmotions_word(unique_strings, llm)
    end_llm_process_time = time.perf_counter()
    elapsed_llm_process_time = end_llm_process_time - start_llm_process_time
    print(f"LLM process + Validation execution time: {elapsed_llm_process_time:.6f} seconds")    
    ENRICHED_JSON = INPUT_EXCEL+'_enriched_data.json'   
    save_enriched_data(enriched_data, ENRICHED_JSON)