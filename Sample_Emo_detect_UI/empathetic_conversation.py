class EmpatheticConversation:
    def __init__(self):
        self.dialogue_adapter = dialogue_adapter

    def generate_response(self, prompt, history=None):
        """Generate an empathetic response based on the conversation history."""
        if history is None:
            history = []
        # Combine history and prompt for context-aware response
        context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
        full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
        return self.dialogue_adapter.generate(full_prompt)

# Global instance
empathetic_conversation = EmpatheticConversation()