import os
from typing import List
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.llms.openai import OpenAI
from interactive_rag.prepare_data import (
    create_persist_dir,
    get_embedding_model,
    load_index_local,
)
from interactive_rag.session_store import SQLiteChatSessionStore


class InteractiveRAG:
    def __init__(self, embedding_model_name: str, data_path: str, llm_name: str):
        """Initialize the InteractiveRAG system.

        This method sets up the index, LLM, and session store required for the chat application.

        :param embedding_model_name: Name of the embedding model used for retrieval.
        :type embedding_model_name: str
        :param data_path: Path to the directory where data and index files are stored.
        :type data_path: str
        :param llm_name: Name of the LLM (e.g., GPT or Llama) used for generating responses.
        :type llm_name: str
        """
        self.index_root = os.path.join(data_path, "index")
        # Initialize retrieval
        self.init_retrieval(embedding_model_name, self.index_root)
        # Initialize LLM
        self.init_llm(llm_name)
        # Init session
        self.session_store = SQLiteChatSessionStore(db_path=data_path)

    def init_retrieval(self, embedding_model_name: str, index_root: str):
        """Initialize retrieval components for embedding-based search.

        This method prepares the embedding model and loads the document index.

        :param embedding_model_name: Name of the embedding model to be used.
        :type embedding_model_name: str
        :param index_root: Path to the root directory for storing the index.
        :type index_root: str
        """
        persist_dir = create_persist_dir(
            persist_root=index_root, model_name=embedding_model_name
        )
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.index = load_index_local(persist_dir=persist_dir)

    def init_llm(self, llm_name: str):
        """Initialize the large language model (LLM) for chat generation.

        Depending on the provided name, this method sets up either an OpenAI GPT model
        or a Llama model.

        :param llm_name: Name of the LLM to be used.
        :type llm_name: str
        :raises ValueError: If the provided LLM name is not supported.
        """
        # TODO hardcoded values should go to config or env
        if "gpt" in llm_name:
            self.llm = OpenAI(model="gpt-4o-mini")
        elif "llama" in llm_name:
            model_url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
            self.llm = LlamaCPP(
                model_url=model_url,
                temperature=0.1,
                max_new_tokens=1024,
                context_window=4096,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": 1},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
        else:
            ValueError(
                "Value for [llm_name] is invalid. Only GPT or LLama models are supported."
            )

    def load_or_create_session(self, user_id: str) -> str:
        """Retrieve an active session for a user or create a new one.

        This method checks for an existing session for the given user and creates a new
        session if none exists.

        :param user_id: The unique identifier for the user.
        :type user_id: str
        :return: The session ID for the user.
        :rtype: str
        """
        session_id = self.session_store.get_active_session_by_user(user_id)
        if session_id is None:
            session_id = self.session_store.create_session(user_id)
        return session_id

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Retrieve the chat history for a given session.

        Converts stored session messages into `ChatMessage` objects for use in chat engines.

        :param session_id: The unique session ID to fetch messages for.
        :type session_id: str
        :return: A list of chat messages associated with the session.
        :rtype: List[ChatMessage]
        """
        messages = self.session_store.get_messages(session_id)
        chat_history = [
            ChatMessage(
                role=(
                    MessageRole.USER
                    if message["sender"] == "user"
                    else MessageRole.ASSISTANT
                ),
                content=message["content"],
            )
            for message in messages
        ]
        return chat_history

    def delete_history(self, user_id: str):
        """Delete chat history for a user.

        This method deletes the active session and its associated messages for the given user.

        :param user_id: The unique identifier for the user.
        :type user_id: str
        """
        self.session_store.delete_session(
            self.session_store.get_active_session_by_user(user_id)
        )

    def chat(self, user_id: str, message: str):
        """Generate a response to a user's message and update chat history.

        This method uses a chat engine to process the user's input, generate a response,
        and update the session with the new messages.

        :param user_id: The unique identifier for the user.
        :type user_id: str
        :param message: The user's input message to process.
        :type message: str
        """
        # Initialize query engine
        query_engine = self.index.as_query_engine(llm=self.llm)
        # Obtain session ID for user.
        session_id = self.load_or_create_session(user_id)
        # Get previous messages in user session
        chat_history = self.get_chat_history(session_id)
        # Initialize chat engine
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            chat_history=chat_history,
            verbose=True,
        )
        response = chat_engine.chat(message)
        # Add message and response to session store
        self.session_store.add_message(session_id, sender="user", content=message)
        self.session_store.add_message(
            session_id, sender="assistant", content=response.response
        )
        print(f"\nAssistant response: {response}")
