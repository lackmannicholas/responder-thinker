"""
Knowledge Thinker — generalist fallback for anything that doesn't fit a domain.

Uses gpt-4.1 (advanced model) because general Q&A requires broad reasoning.
No domain-specific tools — relies on the model's parametric knowledge.
Uses conversation context and user summary to ground answers.
"""

from langsmith import traceable

from backend.config import settings, make_openai_client
from backend.thinkers.base import BaseThinker
from backend.state.user_context import ThinkResult, UserContext

client = make_openai_client()

KNOWLEDGE_SYSTEM_PROMPT = """\
You are a knowledgeable general assistant. Answer the user's question clearly \
and concisely in a way that works well when spoken aloud.

Keep your response to 3-4 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, markdown, or formatting. Just natural speech.
If you don't know something, say so directly rather than guessing.
"""


class KnowledgeThinker(BaseThinker):
    domain = "knowledge"
    description = "General Q&A and factual questions (fallback)"
    model = settings.thinker_model_advanced

    @traceable(name="knowledge_thinker.think")
    async def think(self, query: str, context: list[dict], user_context: UserContext) -> ThinkResult:
        messages: list[dict] = [{"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT}]

        # Include conversation summary for richer grounding
        if user_context.summary.text:
            messages.append({"role": "system", "content": f"Conversation summary so far: {user_context.summary.text}"})

        # Include recent conversation turns to ground the answer in context
        for turn in context[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": query})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return ThinkResult(response=response.choices[0].message.content or "I don't have information on that right now.")
