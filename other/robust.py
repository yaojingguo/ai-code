import pandas as pd

data_path = "/Users/jing/code/github/ai/openai-cookbook/examples/data"
claim_df = pd.read_json(f'{data_path}/scifact_claims.jsonl', lines=True)
claim_df.head()

def build_prompt(claim):
    return [
        {"role": "system", "content": "I will ask you to assess a scientific claim. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence."},
        {"role": "user", "content": f"""
Example:

Claim:
0-dimensional biomaterials show inductive properties.

Assessment:
False

Claim:
1/2000 in UK have abnormal PrP positivity.

Assessment:
True

Claim:
Aspirin inhibits the production of PGE2.

Assessment:
False

End of examples. Assess the following claim:

Claim:
{claim}

Assessment:
"""}
    ]


def assess_claims(claims):
    responses = []
    # Query the OpenAI API
    for claim in claims:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=build_prompt(claim),
            max_tokens=3,
        )
        # Strip any punctuation or whitespace from the response
        responses.append(response.choices[0].message.content.strip('., '))
    return responses

def get_groundtruth(evidence):
    groundtruth = []
    for e in evidence:
        # Evidence is empty 
        if len(e) == 0:
            groundtruth.append('NEE')
        else:
            # In this dataset, all evidence for a given claim is consistent, either SUPPORT or CONTRADICT
            if list(e.values())[0][0]['label'] == 'SUPPORT':
                groundtruth.append('True')
            else:
                groundtruth.append('False')
    return groundtruth


def confusion_matrix(inferred, groundtruth):
    assert len(inferred) == len(groundtruth)
    confusion = {
        'True': {'True': 0, 'False': 0, 'NEE': 0},
        'False': {'True': 0, 'False': 0, 'NEE': 0},
        'NEE': {'True': 0, 'False': 0, 'NEE': 0},
    }
    for i, g in zip(inferred, groundtruth):
        confusion[i][g] += 1
    # Pretty print the confusion matrix
    print('\tGroundtruth')
    print('\tTrue\tFalse\tNEE')
    for i in confusion:
        print(i, end='\t')
        for g in confusion[i]:
            print(confusion[i][g], end='\t')
        print()
    return confusion


corpus_df = pd.read_json(f'{data_path}/scifact_corpus.jsonl', lines=True)
corpus_df.head()


import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
scifact_corpus_collection = chroma_client.create_collection(name='scifact_corpus', embedding_function=embedding_function)

batch_size = 100

for i in range(0, len(corpus_df), batch_size):
    batch_df = corpus_df[i:i+batch_size]
    scifact_corpus_collection.add(
        ids=batch_df['doc_id'].apply(lambda x: str(x)).tolist(), # Chroma takes string IDs.
        documents=(batch_df['title'] + '. ' + batch_df['abstract'].apply(lambda x: ' '.join(x))).to_list(), # We concatenate the title and abstract.
        metadatas=[{"structured": structured} for structured in batch_df['structured'].to_list()] # We also store the metadata, though we don't use it in this example.
    )

def build_prompt_with_context(claim, context):
    return [{'role': 'system', 'content': "I will ask you to assess whether a particular scientific claim, based on evidence provided. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence."}, 
            {'role': 'user', 'content': f""""
The evidence is the following:

{' '.join(context)}

Assess the following claim on the basis of the evidence. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence. Do not output any other text. 

Claim:
{claim}

Assessment:
"""}]


def assess_claims_with_context(claims, contexts):
    responses = []
    # Query the OpenAI API
    for claim, context in zip(claims, contexts):
        # If no evidence is provided, return NEE
        if len(context) == 0:
            responses.append('NEE')
            continue
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=build_prompt_with_context(claim=claim, context=context),
            max_tokens=3,
        )
        # Strip any punctuation or whitespace from the response
        responses.append(response.choices[0].message.content.strip('., '))
    return responses


for idx, val in enumerate(gpt_with_context_evaluation):
    if val == "False. The":
        print(f"{idx}: {val}")



def build_hallucination_prompt(claim):
    return [{'role': 'system', 'content': """I will ask you to write an abstract for a scientific paper which supports or refutes a given claim. It should be written in scientific language, include a title. Output only one abstract, then stop.
    
    An Example:

    Claim:
    A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.

    Abstract:
    BACKGROUND The heritable haemoglobinopathy alpha(+)-thalassaemia is caused by the reduced synthesis of alpha-globin chains that form part of normal adult haemoglobin (Hb). Individuals homozygous for alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count. Alpha(+)-thalassaemia homozygosity confers considerable protection against severe malaria, including severe malarial anaemia (SMA) (Hb concentration < 50 g/l), but does not influence parasite count. We tested the hypothesis that the erythrocyte indices associated with alpha(+)-thalassaemia homozygosity provide a haematological benefit during acute malaria.   
    METHODS AND FINDINGS Data from children living on the north coast of Papua New Guinea who had participated in a case-control study of the protection afforded by alpha(+)-thalassaemia against severe malaria were reanalysed to assess the genotype-specific reduction in erythrocyte count and Hb levels associated with acute malarial disease. We observed a reduction in median erythrocyte count of approximately 1.5 x 10(12)/l in all children with acute falciparum malaria relative to values in community children (p < 0.001). We developed a simple mathematical model of the linear relationship between Hb concentration and erythrocyte count. This model predicted that children homozygous for alpha(+)-thalassaemia lose less Hb than children of normal genotype for a reduction in erythrocyte count of >1.1 x 10(12)/l as a result of the reduced mean cell Hb in homozygous alpha(+)-thalassaemia. In addition, children homozygous for alpha(+)-thalassaemia require a 10% greater reduction in erythrocyte count than children of normal genotype (p = 0.02) for Hb concentration to fall to 50 g/l, the cutoff for SMA. We estimated that the haematological profile in children homozygous for alpha(+)-thalassaemia reduces the risk of SMA during acute malaria compared to children of normal genotype (relative risk 0.52; 95% confidence interval [CI] 0.24-1.12, p = 0.09).   
    CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for alpha(+)-thalassaemia may contribute substantially to their protection against SMA. A lower concentration of Hb per erythrocyte and a larger population of erythrocytes may be a biologically advantageous strategy against the significant reduction in erythrocyte count that occurs during acute infection with the malaria parasite Plasmodium falciparum. This haematological profile may reduce the risk of anaemia by other Plasmodium species, as well as other causes of anaemia. Other host polymorphisms that induce an increased erythrocyte count and microcytosis may confer a similar advantage.

    End of example. 
    
    """}, {'role': 'user', 'content': f""""
    Perform the task for the following claim.

    Claim:
    {claim}

    Abstract:
    """}]


hallucinated_query_result = scifact_corpus_collection.query(query_texts=hallucinated_evidence, include=['documents', 'distances'], n_results=3)

hallucinated_query_result = scifact_corpus_collection.query(query_texts=hallucinated_evidence, include=['documents', 'distances'], n_results=3)
filtered_hallucinated_query_result = filter_query_result(hallucinated_query_result)

gpt_with_hallucinated_context_evaluation = ['True', 'True', 'True', 'NEE', 'True', 'False', 'False', 'True', 'NEE', 'True', 'NEE', 'NEE', 'NEE', 'False', 'NEE', 'False', 'False', 'NEE', 'False', 'True', 'True', 'True', 'NEE', 'NEE', 'True', 'True', 'NEE', 'True', 'True', 'True', 'True', 'NEE', 'NEE', 'NEE', 'True', 'NEE', 'NEE', 'True', 'False', 'True', 'True', 'True', 'False', 'True', 'True', 'NEE', 'NEE', 'True', 'True', 'NEE']
