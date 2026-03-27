import pandas as pd
from typing import List, Dict, Any
import argparse
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import json



def read_excel_to_list_of_dicts(file_path: str) -> List[Dict[str, Any]]:
    
    TARGET_COLUMNS = [
        "ID",
        "Genere", 
        "Età", 
        "Istruzione", 
        "Ruolo", 
        "Anni esp", 
        "Ambito", 
        "Grado", 
        "SSSG", 
        "Regione"
    ]
    
    try:
        
        
        df = pd.read_excel(file_path, usecols=TARGET_COLUMNS, dtype=str, header=0)
        df = df.fillna("")
        
        
        data_list = df.to_dict('records')
        
        return data_list
        
    except FileNotFoundError:
        print(f"Error: The file was not found at path: {file_path}")
        return []
    except ValueError as e:
        
        if "Usecols do not match columns" in str(e):
            print(f"Error: One or more specified columns are missing from the Excel file.")
            print(f"Required columns: {TARGET_COLUMNS}")
        else:
            print(f"An error occurred while reading the Excel file: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
    


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
    print("\nModel loading complete. Please review the output above this line for details on GPU offloading.")
    print("Look for lines like 'llm_load_tensors: offloading X/Y layers to GPU'.")
    print("If no such lines appear, or if X is 0, then no layers were offloaded to the GPU (or you set n_gpu_layers=0).")

    return llm



def llm_enrich(profilo: Dict[str, str], llm: Llama) -> List[Dict[str, object]]:

    (genere, eta, istruzione, ruolo, anni_esp, ambito, grado, sssg, regione) = (
        profilo.get('Genere', ""), 
        profilo.get('Età', ""), 
        profilo.get('Istruzione', ""), 
        profilo.get('Ruolo', ""), 
        profilo.get('Anni esp', ""), 
        profilo.get('Ambito', ""), 
        profilo.get('Grado', ""), 
        profilo.get('SSSG', ""), 
        profilo.get('Regione', "")
    )


    promptRaw = f'''
[CONTESTO]
Sei un modello linguistico incaricato di simulare un partecipante umano plausibile.
Ti verrà fornito un profilo anonimizzato con informazioni demografiche e professionali.
Il tuo compito è rispondere a un questionario come se fossi veramente una persona con quel 
profilo, adottando linguaggio, tono, valori, atteggiamenti e livello culturale compatibili
con il profilo fornito.
Tieni presente che il questionario verrà compilato tramite un sito web.
In base alla domanda, certe risposte richiederanno di selezionare una tra più opzioni 
fornite (domande q17–q19), altre risposte richiederanno invece di scrivere un testo libero
(domande q1 e q2) o scrivere liberamente una parola (q3–q16).
Le risposte dovranno essere compatibili con la compilazione in un contesto digitale 
(e.g., computer, smartphone o tablet), con tutte le caratteristiche che questo 
può comportare (es. risposte più sintetiche, uso di abbreviazioni, errori di battitura, 
ecc.).


[ISTRUZIONI]
1. Leggi con attenzione il profilo della persona riportato più sotto.
2. Rispondi alle domande del questionario immedesimandoti pienamente in questa persona.
3. Rispondi a TUTTE le domande del questionario.
4. Rispetta queste regole formali:
   • q1 e q2 → risposte libere, di lunghezza compatibile con la compilazione di un questionario online.
   • q3–q16 → UNA SOLA parola, preferibilmente un aggettivo.
   • q17–q19 → scegli UNA SOLA delle opzioni fornite e copia il testo completo della risposta, compreso il codice iniziale della risposta
5. Non fornire spiegazioni, non commentare le scelte, non uscire mai dal ruolo.
6. Scrivi le risposte in modo coerente con il profilo indicato.
7. **Ritorna SOLO le risposte nel formato JSON**, senza alcun testo aggiuntivo (come introduzioni o spiegazioni) o blocchi di codice markdown (```json):
{ 
	"q1" : "testo risposta q1",
	"q2" : "testo risposta q2",
	"q3" : "parola risposta q3",
	...
} 

[PROFILO DA IMPERSONARE]

Genere: {genere}
Età: {eta}
Regione italiana in cui lavori e vivi: {regione}
Livello di istruzione: {istruzione}
Ruolo nei confronti della scuola: {ruolo}
'''

    if ruolo in ["Insegnante di sostegno", "Insegnante curricolare"]:    
        promptRaw += f'''
        Anni di lavoro a scuola: {anni_esp}
        Ambito disciplinare: {ambito}
        Grado scolastico: {grado}
        '''
    if grado == "Secondaria di II grado":
        promptRaw += f'''
            Scuola secondaria: {sssg}
        '''
    
    promptRaw += f'''

[QUESTIONARIO]
Stiamo raccogliendo esperienze, vissuti, riflessioni e testimonianze sui percorsi di inclusione scolastica: i punti di forza, le criticità, le storie quotidiane che rendono la scuola un luogo davvero aperto a tutte e tutti.
In vista del Convegno “La Qualità dell’inclusione scolastica e sociale”, ti invitiamo a partecipare a questo questionario. 

q1. Ricorda una situazione in cui hai vissuto una piena inclusione a scuola. Che cosa ha funzionato bene?
q2. Pensa alla tua specifica situazione dell'ultimo anno di scuola, che ostacoli hai incontrato per l’inclusione?

Pensando alla tua esperienza di inclusione, scrivi la prima parola (preferibilmente un aggettivo) che ti viene in mente se leggi: 

q3. Aula di sostegno
q4. PEI
q5. Insegnante di sostegno
q6. Co-docenza/collaborazione curricolare-sostegno
q7. Gruppo classe
q8. Personalizzazione didattica
q9. Educatori/assistenti autonomia e comunicazione
q10. Famiglia dell'alunno/a con disabilità
q11. Progetto di Vita
q12. Collaborazione coi servizi sanitari
q13. Autodeterminazione
q14. GLO
q15. Valutazione (dell’alunno/a con disabilità)
q16. Dirigente scolastico/a

q17. Sarebbe opportuno che gli alunni con disabilità, in base al grado della loro difficoltà, avessero la possibilità di essere inseriti in contesti scolastici diversi (es. modello a tre vie): scuole solo per alunni con disabilità (casi di disabilità grave), classi per alunni con disabilità nelle scuole normali (casi di disabilità media), inclusione piena in classe (casi di disabilità lieve).
A: Per niente d'accordo
B: Abbastanza in disaccordo
C: Un po' d'accordo
D: Completamente d'accordo

'''
    if ruolo in ["Insegnante di sostegno", "Insegnante curricolare"]:   
        promptRaw += f'''

q18. Le difficoltà quotidiane nel lavorare in classe con un alunno con grave disabilità, mi portano a credere che in alcuni casi l’inclusione non sia la scelta migliore.
A: Per niente d'accordo
B: Abbastanza in disaccordo
C: Un po' d'accordo
D: Completamente d'accordo

q19. Nel lavoro quotidiano con un alunno con disabilità grave, mi è capitato spesso di pensare che una vera inclusione non fosse fattibile.
A: Per niente d'accordo
B: Abbastanza in disaccordo
C: Un po' d'accordo
D: Completamente d'accordo

    '''
    
    
    prompt = f"""<start_of_turn>user
    {promptRaw}

    <end_of_turn>
    <start_of_turn>model
    """

    TEMPERATURE=1.0

    output = llm(
        prompt,
        max_tokens=5000,
        temperature=TEMPERATURE,
        top_p=0.95,
        echo=False,
        stream=False
    )

    enriched = output["choices"][0]["text"].replace("```json", "").replace("```","").strip()
    
    try:
        print("#"*20)
        print(f"Profile: '{profilo}'")
        
        enriched = json.loads(enriched)
        
        if ruolo not in ["Insegnante di sostegno", "Insegnante curricolare"]:   
            enriched["q18"] = "e: vuota"
            enriched["q19"] = "e: vuota"
        print(f"Output: '{enriched}'")
        print("#"*20)
        return enriched
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON string: {e}")
        return {}



def incremental_save(row_data: Dict[str, Any], output_file_path: str, is_first_row: bool):
    
    
    OUTPUT_COLUMNS = [
        "ID","Genere", "Età", "Istruzione", "Ruolo", "Anni esp", "Ambito", 
        "Grado", "SSSG", "Regione", "Positiva", "Negativa", 
        "Sentiment parole_1: - Aula di sostegno",
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
        "Sentiment parole_14: - Dirigente scolastico/a", 
        "Tre vie", "Non migliore", "Non fattibile"
    ]
    
    
    COLUMN_MAPPING = {
        "q1": "Positiva",
        "q2": "Negativa",
        "q3": "Sentiment parole_1: - Aula di sostegno",
        "q4": "Sentiment parole_2: - PEI",
        "q5": "Sentiment parole_3: - Insegnante di sostegno",
        "q6": "Sentiment parole_4: - Co-docenza/collaborazione curricolare-sostegno",
        "q7": "Sentiment parole_5: - Gruppo classe",
        "q8": "Sentiment parole_6: - Personalizzazione didattica",
        "q9": "Sentiment parole_7: - Educatori/assistenti autonomia e comunicazione",
        "q10": "Sentiment parole_8: - Famiglia dell'alunno/a con disabilità",
        "q11": "Sentiment parole_9: - Progetto di Vita",
        "q12": "Sentiment parole_10: - Collaborazione coi servizi sanitari",
        "q13": "Sentiment parole_11: - Autodeterminazione",
        "q14": "Sentiment parole_12: - GLO",
        "q15": "Sentiment parole_13: - Valutazione (dell’alunno/a con disabilità)",
        "q16": "Sentiment parole_14: - Dirigente scolastico/a",
        "q17": "Tre vie",
        "q18": "Non migliore",
        "q19": "Non fattibile",
    }
    
    
    final_row = {col: "" for col in OUTPUT_COLUMNS}
    
    
    final_row.update(row_data.get('profilo', {}))

    
    enriched_data = row_data.get('enriched', {})
    for q_key, col_name in COLUMN_MAPPING.items():
        
        
        
        
        final_row[col_name] = enriched_data.get(q_key, "")

    
    df_new_row = pd.DataFrame([final_row], columns=OUTPUT_COLUMNS)

    
    try:
        if is_first_row:
            
            df_new_row.to_excel(output_file_path, index=False)
        else:
            
            
            
            
            
            df_existing = pd.read_excel(output_file_path)
            
            
            df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
            
            
            df_combined.to_excel(output_file_path, index=False)
            
        print(f"Row successfully saved incrementally to {output_file_path}")

    except Exception as e:
        print(f"\n❌ An error occurred during incremental save: {e}")


def validate_llm_output(output_dict: Dict[str, str]) -> bool:

    
    required_keys = [f"q{i}" for i in range(1, 20)]
    if not all(key in output_dict for key in required_keys):
        print(f"Validation FAILED: Missing one or more required keys (q1-q19).")
        return False

    
    
    
    for key in required_keys:
        value = output_dict.get(key)
        
        
        if not isinstance(value, str) or value.strip() == "":
            print(f"Validation FAILED: Value for key '{key}' is empty or not a string.")
            return False
    

    



    
    allowed_q17_q18_q19 = {
        "a: per niente d'accordo",
        "b: abbastanza in disaccordo",
        "c: un po' d'accordo",
        "d: completamente d'accordo",
        "e: vuota"
    }

    
    def is_valid_response(key, allowed_set):
        
        
        value = str(output_dict.get(key)).strip().lower() 

        
        if value in allowed_set:
            return True

        
        
        code = value.split(' ')[0].rstrip(':').rstrip('.')

        
        if any(option.startswith(code) for option in allowed_set):
            return True

        return False

    
    if not is_valid_response('q17', allowed_q17_q18_q19):
        print(f"Validation FAILED: q17 response ('{output_dict.get('q17')}') is not a valid option.")
        return False

    
    if not is_valid_response('q18', allowed_q17_q18_q19):
        print(f"Validation FAILED: q18 response ('{output_dict.get('q18')}') is not a valid option.")
        return False

    if not is_valid_response('q19', allowed_q17_q18_q19):
        print(f"Validation FAILED: q19 response ('{output_dict.get('q19')}') is not a valid option.")
        return False

    return True



if __name__ == "__main__":
    
    MAX_RETRIES = 3
    
    total_failed_profiles = 0
    total_llm_attempts = 0
    total_validation_failures = 0 

    parser = argparse.ArgumentParser(description="Extract and enrich data from an Excel file using an LLM, then save the results.")
    
    
    parser.add_argument(
        "file_name", 
        nargs="?", 
        default="/mnt/sharedHD/CodeRepo/simulate_erickson/data/machines_gemma/Q2025_6ottobre2025_finalCleaned.xlsx", 
        type=str, 
        help="this is the localname of the input file (without path) containing the excel data."
    )

    args = parser.parse_args()
    file_name = args.file_name
    
    output_file_name = file_name.replace(".xlsx",".gemma_enriched_output.xlsx")

    profili = read_excel_to_list_of_dicts(file_name)
    print(f"Loaded {len(profili)} profiles.")

    
    try:
        llm = load_model()
    except Exception as e:
        print(f"❌ Error loading LLM model: {e}")
        exit(1)
    
    is_first_row = True 

    
    for i, profilo in enumerate(profili):
        print(f"\n--- Processing profile {i+1}/{len(profili)} --- (Failed profiles so far: {total_failed_profiles} - Failed validations so far: {total_validation_failures})")
        
        enriched_output = {}
        attempt = 0
        
        while attempt < MAX_RETRIES:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} for profile {i+1}.")
            
            total_llm_attempts += 1 
            
            
            temp_output = llm_enrich(profilo, llm)
            
            
            if validate_llm_output(temp_output):
                enriched_output = temp_output
                print(f"Profile {i+1} successfully processed and validated on attempt {attempt + 1}.")
                break 
            
            
            total_validation_failures += 1
            print(f"Validation failed on attempt {attempt + 1}. Retrying...")
            
            attempt += 1
        
        
        if not enriched_output:
            print(f"❌ Failed to get valid output for profile {i+1} after {MAX_RETRIES} attempts. Saving empty data for enriched fields.")
            total_failed_profiles += 1

        
        q17_val = enriched_output.get("q17", "")
        q18_val = enriched_output.get("q18", "")
        q19_val = enriched_output.get("q19", "")
        
        enriched_output["q17"] = q17_val[3:] if len(q17_val) > 3 else q17_val
        enriched_output["q18"] = q18_val[3:] if len(q18_val) > 3 else q18_val
        enriched_output["q19"] = q19_val[3:] if len(q19_val) > 3 else q19_val

        enriched_output["q18"] = enriched_output["q18"].replace("vuota", "").strip()
        enriched_output["q19"] = enriched_output["q19"].replace("vuota", "").strip()

        
        combined_data = {
            'profilo': profilo,
            'enriched': enriched_output
        }
        
        
        incremental_save(combined_data, output_file_name, is_first_row)
        is_first_row = False
        
    
    total_profiles = len(profili)
    successful_profiles = total_profiles - total_failed_profiles
    
    
    validation_success_rate = (total_llm_attempts - total_validation_failures) * 100 / total_llm_attempts if total_llm_attempts > 0 else 0
    
    print("\n" + "="*50)
    print("✨ Processing Summary ✨")
    print(f"Total profiles processed: {total_profiles}")
    print(f"Total LLM attempts made: {total_llm_attempts}")
    print(f"Total validation failures (including retries): {total_validation_failures}")
    print(f"Raw Validation Success Rate: {validation_success_rate:.2f}%")
    print(f"✅ Profiles successfully validated: {successful_profiles}")
    print(f"❌ Profiles failed (after {MAX_RETRIES} retries): {total_failed_profiles}")
    print(f"Output saved to: {output_file_name}")
    print("="*50)