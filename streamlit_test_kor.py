import streamlit as st
from transformers import ElectraModel,AutoModelForTokenClassification
from transformers import TokenClassificationPipeline 
from annotated_text import annotated_text
from tokenization_kocharelectra import KoCharElectraTokenizer 
st.header("NER Demo")

text = st.text_area('Enter text:', value="미국·일본·호주·인도가 참여한 쿼드(Quad) 4개국이 24일 일본에서 정상회의를 열고 인도·태평양에서 세력을 확장하는 중국을 견제하는 방안 등을 논의했다. 조 바이든 미국 대통령, 기시다 후미오 일본 총리, 나렌드라 모디 인도 총리, 앤서니 앨버니지 호주 총리는 이날 도쿄 총리관저에서 쿼드 정상회의를 열었다. 쿼드 4국 정상이 대면으로 회의를 하는 것은 작년 9월 24일 미국 워싱턴DC에서 회의를 개최한 후 약 8개월 만이다. 바이든 대통령은 회의 모두발언에서 \"인도적 재앙을 촉발한 러시아가 우크라이나 문화를 지워버리려 하고 있다\"며 \"미국은 국제적 대응을 위해 파트너들과 계속 협력할 것\"이라고 말했다.")
@st.cache
def load_model():
    tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')
    model = AutoModelForTokenClassification.from_pretrained('monologg/kocharelectra-base-modu-ner-all')
    nlp = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer)
    return nlp
#nlp = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple" )


colour_map = {
  'PS': '#8fbc8f',
  'DT': '#b0c4de',
  'TI': '#f54f07',
  'AM': '#f54f07',
  'PT': '#dd07f5',
  'FD': '#0721f5',
  'TR': '#07f57a',
  'MT': '#f507bf',
  'TM': '#f5072e',
  'AF': '#f5a907',
  'LC': '#ff7373',
  'QT': '#3c00ff',
  'OG': '#750ed7',
  'EV': '#0081f5',
  'CV': '#fc8407',
}


nlp=load_model()
if text:
  ner_results = nlp(text)
  new_entities=[]
  entities={}
  for n in ner_results:
    if "B" in n["entity"]:
        if entities :
            new_entities.append(entities)
            entities={}
        entities['entity_group']=n["entity"][2:]
        entities['word']=n['word']
        entities['start']=int(n['index'])-1
        entities['end']=int(n['index'])-1
    elif "I" in n["entity"]:
        entities['word']+=n['word']
        entities['end']+=1
  s = 0
  parsed_text = []
  for n in new_entities:

    parsed_text.append(text[s:n["start"]])
    parsed_text.append((n["word"], n["entity_group"], colour_map[n["entity_group"]]))
    s = n["end"]+1
  parsed_text.append(text[s:])
  annotated_text(*parsed_text)
  st.json(new_entities)


