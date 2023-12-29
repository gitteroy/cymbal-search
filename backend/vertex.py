#@title VertexAI

import re
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

def vertex_qa(
  project_id,
  llmModel,
  query,
  summary,
  checkedItems,
  searchResults,
  userInput,
  temperature,
  topK,
  topP
  ):

  ######################################################################################################

  """ Snippet / Answer / Segment """
  snippet_pattern = r"\{\{snippet_(\d+)\}\}"
  answer_pattern = r"\{\{answer_(\d+)-(\d+)\}\}"
  segment_pattern = r"\{\{segment_(\d+)-(\d+)\}\}"
  doc_pattern = r"\{\{docName_(\d+)\}\}"

  # Find all occurrences of each item in the userInput
  snippets = re.findall(snippet_pattern, userInput)
  answers = re.findall(answer_pattern, userInput)
  segments = re.findall(segment_pattern, userInput)
  docs = re.findall(doc_pattern, userInput)

  # Create a dictionary to store the key-value pairs
  result_dict = {}

  # Process snippet items and store in the result_dict
  for snippet_index in snippets:
    snippet_index = int(snippet_index)
    if snippet_index <= len(searchResults):
      # print("Snippet:",snippet_index)
      value = searchResults[snippet_index - 1]["snippets"][0]
    else:
      value = ""
    key = f"{{snippet_{snippet_index}}}"
    result_dict[key] = value

  # Process answer items and store in the result_dict
  for answer_index_1, answer_index_2 in answers:
    answer_index_1, answer_index_2 = int(answer_index_1), int(answer_index_2)
    if answer_index_1 <= len(searchResults) and answer_index_2 <= len(searchResults[answer_index_1 - 1]["extractive_answers_content"]):
      # print("Answer:", answer_index_1, answer_index_2)
      value = searchResults[answer_index_1 - 1]["extractive_answers_content"][answer_index_2 - 1]
    else:
      value = ""
    key = f"{{answer_{answer_index_1}-{answer_index_2}}}"
    result_dict[key] = value

  # Process answer items and store in the result_dict
  for segment_index_1, segment_index_2 in segments:
    segment_index_1, segment_index_2 = int(segment_index_1), int(segment_index_2)
    if segment_index_1 <= len(searchResults) and segment_index_2 <= len(searchResults[segment_index_1 - 1]["extractive_segments"]):
      # print("Segment:", segment_index_1, segment_index_2)
      value = searchResults[segment_index_1 - 1]["extractive_segments"][segment_index_2 - 1]
    else:
      value = ""
    key = f"{{segment_{segment_index_1}-{segment_index_2}}}"
    result_dict[key] = value

  # Process doc items and store in the result_dict
  for doc_index in docs:
    doc_index = int(doc_index)
    if doc_index <= len(searchResults):
      # print("Doc:", doc_index)
      value = searchResults[doc_index - 1]["filter_name"]
    else:
      value = ""
    key = f"{{docName_{doc_index}}}"
    result_dict[key] = value

  # Process other {{}} and store in the result_dict
  result_dict["{{query}}"] = query
  result_dict["{{summary}}"] = summary
  result_dict["{{checked_snippets}}"] = ','.join(str(item['snippets']) for item in checkedItems)
  result_dict["{{checked_answers}}"] = ','.join(str(item['extractive_answers_content']) for item in checkedItems)
  result_dict["{{checked_segments}}"] = ','.join(str(item['extractive_segments']) for item in checkedItems)

  ######################################################################################################

  """ Vertex AI """

  # https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models
  max_output_tokens = 1024 # text-bison
  if llmModel=="text-bison-32k":
    max_output_tokens = 8192 # max output for text-bison-32k

  temperature = 0.0 if temperature<0.0 else temperature
  temperature = 1.0 if temperature>1.0 else temperature
  topK = 1 if topK<1 else topK
  topK = 40 if topK>40 else topK
  topP = 0.0 if topP<0.0 else topP
  topP = 1.0 if topP>1.0 else topP

  ## Replace each {x} with its actual value
  for i in result_dict:
    userInput = userInput.replace(i, result_dict[i])

  # result = llm(userInput)
  result = predict_large_language_model_sample(project_id, llmModel, temperature, topP, topK, userInput)

  print("Final Prompt:\n", userInput)
  # print("-----------")
  print("LLM Output:\n", result)

  return result

def predict_large_language_model_sample(
  project_id: str,
  model_name: str,
  temperature: float,
  top_p: float,
  top_k: int,
  content: str,
  max_decode_steps: int=256,
  location: str = "us-central1",
  tuned_model_name: str = "",
  ) :
  """Predict using a Large Language Model."""
  
  ## Text Models
  if "gemini" not in model_name:
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,
    )

  ## Gemini
  else:
    from google.cloud import aiplatform
    aiplatform.init(project=project_id)
    gemini_pro_model = GenerativeModel("gemini-pro")
    # gemini_pro_vision_model = GenerativeModel("gemini-pro-vision")
    model_response = gemini_pro_model.generate_content(content)
    response = model_response.candidates[0].content.parts[0]

  return response.text