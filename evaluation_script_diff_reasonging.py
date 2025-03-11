import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from bert_score import score
from nltk.translate import meteor
from nltk.tokenize import word_tokenize

def get_openai_response(prompt, model):
    api_key = "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"
    org_key = "org-ewbrRzXdrHxv7hV0WyCFzGdD"

    """
    Sends a prompt to OpenAI's API and retrieves the response.

    Args:
        prompt (str): The prompt for the LLM.
        model (str): The model to use (e.g., 'gpt-4').
        api_key (str): OpenAI API key.
        org_key (str): OpenAI organization key.

    Returns:
        str: The response text from the LLM.
    """
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Organization': org_key
    }
    data = {
        'messages': [{'role': 'system', 'content': prompt}],
        'model': model,
        'temperature': 0.0
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def llm_as_a_judge_prompt(conversation, reasoning_summary):
    """
    Generates a prompt for the LLM-as-a-judge evaluation.

    Args:
        conversation (str): The conversation or context text.
        reasoning_summary (str): The reasoning summary text.

    Returns:
        str: The generated prompt.
    """
    prompt = f"""### Instruction: Evaluate the reasoning for predicting the difficulty of medical questionnaires based on the conversation.

### Scoring Criteria:

**Case 1: Simple Questionnaire (Low Difficulty)**
- **2 points** if the reasoning clearly indicates that the questionnaire is simple with few questions, minimal medical history required, or if the conversation suggests an easy-to-understand questionnaire.
- **1 point** if the reasoning indicates that the questionnaire might be simple, but lacks clarity or supporting evidence from the conversation.
- **0 points** if no reasoning is provided or it contradicts the idea of simplicity.

**Case 2: Complex Questionnaire (High Difficulty)**
- **2 points** if the reasoning clearly indicates that the questionnaire is complex with multiple questions, detailed medical history required, or if the conversation suggests a high level of detail needed from the patient.
- **1 point** if the reasoning indicates that the questionnaire might be complex, but lacks enough supporting evidence from the conversation.
- **0 points** if no reasoning is provided or it contradicts the idea of complexity.

**General Evaluation Criteria:**
- **Clarity and Coherence**: 0.5 points for clear, well-structured reasoning.
- **Relevance**: 0.5 points if the reasoning is relevant to predicting the difficulty of the questionnaire based on the conversation.
- **Accuracy**: 1 point if the difficulty prediction aligns with the conversation content.

### Input:
- **Conversation**: 
{conversation}

- **Summary (Reasoning for difficulty prediction)**: 
{reasoning_summary}

### Output:
- "score: <total points>"
- Briefly justify your score, up to 50 words.
"""
    return prompt

def evaluate_difficulty_and_reasoning(df, model):
    """
    Evaluates the dataframe, including `LLM as a judge` metric.

    Args:
        df (pd.DataFrame): Input dataframe with columns for difficulty, predictions, and reasoning.
        model (str): The LLM model to use (e.g., 'gpt-4').
        api_key (str): OpenAI API key.
        org_key (str): OpenAI organization key.

    Returns:
        dict: A dictionary of metrics for difficulty, reasoning, and LLM as a judge.
    """
    # Validate required columns
    required_columns = ['difficulty', 'difficulty_prediction', 'reasoning', 'reasoning_prediction']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain the following columns: {required_columns}")

    # Metrics for difficulty predictions
    difficulty_metrics = {
        "accuracy": accuracy_score(df['difficulty'], df['difficulty_prediction']),
        "f1": f1_score(df['difficulty'], df['difficulty_prediction'], average='weighted'),
        "precision": precision_score(df['difficulty'], df['difficulty_prediction'], average='weighted'),
        "recall": recall_score(df['difficulty'], df['difficulty_prediction'], average='weighted'),
        "confusion_matrix": confusion_matrix(df['difficulty'], df['difficulty_prediction']).tolist()
    }

    # Metrics for reasoning predictions
    reasoning_metrics = {
        "bert_score": {"precision": [], "recall": [], "f1": []},
        "meteor_score": []
    }
    llm_judge_scores = []
    for _, row in df.iterrows():
        ref = row['reasoning']
        pred = row['reasoning_prediction']
        # Compute BERTScore
        P, R, F1 = score([pred], [ref], lang='en', verbose=False)
        reasoning_metrics["bert_score"]["precision"].append(P.mean().item())
        reasoning_metrics["bert_score"]["recall"].append(R.mean().item())
        reasoning_metrics["bert_score"]["f1"].append(F1.mean().item())

        # Compute METEOR
        meteor_score_value = meteor([word_tokenize(ref)], word_tokenize(pred))
        reasoning_metrics["meteor_score"].append(meteor_score_value)

        # Generate LLM-as-a-judge prompt
        prompt = llm_as_a_judge_prompt(ref, pred)
        try:
            llm_response = get_openai_response(prompt, model)
            llm_score = extract_score_from_llm_response(llm_response)
        except Exception as e:
            print(f"Error in LLM scoring: {e}")
            llm_score = None
        llm_judge_scores.append(llm_score)

    # Aggregate BERTScore and METEOR
    reasoning_metrics["bert_score"] = {
        "precision": np.mean(reasoning_metrics["bert_score"]["precision"]),
        "recall": np.mean(reasoning_metrics["bert_score"]["recall"]),
        "f1": np.mean(reasoning_metrics["bert_score"]["f1"])
    }
    reasoning_metrics["meteor_score"] = np.mean(reasoning_metrics["meteor_score"])

    # Combine all metrics
    return {
        "difficulty_metrics": difficulty_metrics,
        "reasoning_metrics": reasoning_metrics,
        "llm_judge_scores": {
            "mean_score": np.nanmean(llm_judge_scores),
            "scores": llm_judge_scores
        }
    }

def extract_score_from_llm_response(response):
    """
    Extracts the score from LLM response text.

    Args:
        response (str): The text response from the LLM.

    Returns:
        float: The extracted score.
    """
    pattern = r"score:\s*(\d+(\.\d+)?)"
    match = re.search(pattern, response.lower())
    if match:
        return float(match.group(1))
    else:
        return None
