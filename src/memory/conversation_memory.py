from collections import deque


class ConversationMemory:

    def __init__(self, max_messages: int = 6):

        self.messages = deque(maxlen=max_messages)

    def add_user_message(self, message: str):

        self.messages.append(
            {
                "role": "user",
                "content": message
            }
        )

    def add_assistant_message(self, message: str):

        self.messages.append(
            {
                "role": "assistant",
                "content": message
            }
        )

    def get_messages(self):

        return list(self.messages)

    def clear(self):

        self.messages.clear()