from transformers import pipeline

'''
available tasks are ['any-to-any', 'audio-classification', 'automatic-speech-recognition', 'depth-estimation', 
'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 
'image-segmentation', 'image-text-to-text', 'keypoint-matching', 'mask-generation', 'ner', 'object-detection', 
'sentiment-analysis', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 
'text-to-speech', 'token-classification', 'video-classification', 'zero-shot-audio-classification', 'zero-shot-classification', 
'zero-shot-image-classification', 'zero-shot-object-detection']"

'''

def summarize_text_ai(text, max_length=50, min_length=6):
    # Load a pre-trained summarization model (e.g., "t5-small")
    summarizer = pipeline("text-generation", model="t5-small")
    
    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    
    return summary

# Example usage with a long text
long_text = """My experience with AI/ ML contains the following production applications, short descriptions below.
 
	•  Insurance on boarding: Model to score users for auto insurance and worker's compensation customers / businesses using models on SAS, python environment 300k requests/ sec on hybrid cloud (public/ private cloud)
 
	• Financial Fraud Detection: Multi model platform for 'real-time' financial fraud with semi-supervised clustering, graph models with Scala/ spark, Torch, MLOps on AWS, 900k messages/ sec (AWS cloud)
 
	• Asset Security Prediction: Multi model Machine learning and data analysis of image data with Tensor-flow, Python, Scala/ Spark, MLOps (AWS and GCP later ported to Azure)
 
	• Video Analytics, viewer analysis: Content analyses, video ad placement, monthly revenue and classification with open source tools, MLOps over 1M messages (AWS)
 
	• Customer analysis, up-sell, central customer knowledge base Generative AI/LLM/ RAG using ChatGPT and LLaMA initiative for professional services, sales and marketing, scalability 
		○ My major contribution/ impact was scaling the LLM platform to accommodate 1000+ concurrent internal / external users
		○ UI/UX with open source technology stack like StreamLit, LangChain, LlamaIndex etc..
		○ Multi-agent collaborative framework development
	• Reward Model:
Evaluates the quality of the LLM's generated response based on various criteria (e.g., relevance, factual accuracy, fluency, user satisfaction). This model provides a reward signal to the RL agent.
	• RL Agent:
Learns from the reward signal to optimize the generation process. This could involve fine-tuning the LLM's parameters, adjusting the retrieval strategy, or even learning to select the most appropriate retrieved documents. Techniques like Proximal Policy Optimization (PPO) or Reinforcement Learning from Human Feedback (RLHF) are commonly used here.
	• Policy Network (within LLM/RL Agent):
The RL agent learns a policy that maps states (e.g., augmented prompt, retrieved documents) to actions (e.g., generating text, selecting documents).
	• The LLM generates a refined response, iteratively improved through the RL feedback loop, and delivers it to the user.
"""

print(summarize_text_ai(long_text))
