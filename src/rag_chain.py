"""
RAG chain and memory management for RAG Chatbot.

Handles LLM instantiation, conversation memory, and conversational retrieval chain creation.
"""

from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory


def create_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.5,
    top_p: float = 0.95
):
    """
    Create an LLM instance based on provider.
    
    Args:
        provider: LLM provider name ("OpenAI", "Google", "HuggingFace")
        model: Model name
        api_key: API key for the provider
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        
    Returns:
        LLM instance
    """
    if provider == "OpenAI":
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            model_kwargs={"top_p": top_p}
        )
    
    elif provider == "Google":
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True
        )
    
    elif provider == "HuggingFace":
        return HuggingFaceHub(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "max_new_tokens": 1024
            }
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_memory(
    model_name: str,
    llm=None,
    memory_max_token: int = 1024
):
    """
    Create conversation memory.
    
    Uses ConversationSummaryBufferMemory for gpt-3.5-turbo to handle token limits.
    Uses ConversationBufferMemory for other models.
    
    Args:
        model_name: Name of the model
        llm: LLM instance (required for ConversationSummaryBufferMemory)
        memory_max_token: Maximum tokens for summary buffer memory
        
    Returns:
        Memory instance
    """
    if model_name == "gpt-3.5-turbo":
        if llm is None:
            raise ValueError("LLM instance required for ConversationSummaryBufferMemory")
        
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=llm,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
    
    return memory


def get_condense_question_prompt() -> PromptTemplate:
    """
    Get the prompt template for condensing follow-up questions.
    
    Returns:
        PromptTemplate for question condensation
    """
    template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""
    
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )


def get_answer_prompt(language: str = "english") -> ChatPromptTemplate:
    """
    Get the prompt template for answering questions.
    
    Args:
        language: Language for the answer
        
    Returns:
        ChatPromptTemplate for answering
    """
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    
    return ChatPromptTemplate.from_template(template)


def create_conversational_chain(
    retriever,
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.5,
    top_p: float = 0.95,
    language: str = "english",
    memory_max_token: int = 1024,
    chain_type: str = "stuff"
) -> Tuple:
    """
    Create a ConversationalRetrievalChain.
    
    The chain:
    1. Takes a follow-up question and chat history
    2. Rephrases the question into a standalone query
    3. Retrieves relevant documents
    4. Generates an answer using the context and chat history
    
    Args:
        retriever: Retriever instance
        provider: LLM provider name
        model: Model name
        api_key: API key
        temperature: Temperature for response generation
        top_p: Top-p for response generation
        language: Language for responses
        memory_max_token: Max tokens for memory
        chain_type: Chain type ("stuff", "map_reduce", etc.)
        
    Returns:
        Tuple of (chain, memory)
    """
    # Create LLMs
    # Use lower temperature for question condensation
    condense_llm = create_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0.1,
        top_p=0.95
    )
    
    response_llm = create_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p
    )
    
    # Create memory
    # For gpt-3.5-turbo, use a special LLM for memory summarization
    if model == "gpt-3.5-turbo" and provider == "OpenAI":
        memory_llm = ChatOpenAI(
            api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        memory = create_memory(model, llm=memory_llm, memory_max_token=memory_max_token)
    else:
        memory = create_memory(model, memory_max_token=memory_max_token)
    
    # Get prompts
    condense_question_prompt = get_condense_question_prompt()
    answer_prompt = get_answer_prompt(language)
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=condense_llm,
        llm=response_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True
    )
    
    return chain, memory


def format_source_documents(source_documents: list) -> str:
    """
    Format source documents for display.
    
    Args:
        source_documents: List of source documents from chain response
        
    Returns:
        Formatted markdown string
    """
    documents_content = ""
    
    for document in source_documents:
        # Try to get page number if available
        try:
            page = f" (Page: {document.metadata['page']})"
        except (KeyError, AttributeError):
            page = ""
        
        # Get source
        try:
            source = document.metadata.get("source", "Unknown")
        except AttributeError:
            source = "Unknown"
        
        documents_content += f"**Source: {source}{page}**\n\n"
        documents_content += document.page_content + "\n\n\n"
    
    return documents_content
