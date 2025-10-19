import { useMemo, useState } from "react";

import ChatMessage from "../components/ChatMessage";
import { chatWithAssistant } from "../lib/api";

const makeId = () => (crypto?.randomUUID ? crypto.randomUUID() : `id-${Date.now()}-${Math.random()}`);

const initialAssistantMessage = {
  id: "welcome",
  role: "assistant",
  content: "Hi! Ask me about furniture or the vibe you're looking for and I'll curate some pieces.",
};

function ChatPage() {
  const [messages, setMessages] = useState([initialAssistantMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    setError(null);
  const userMessage = { id: makeId(), role: "user", content: trimmed };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    try {
      setLoading(true);
      const response = await chatWithAssistant(nextMessages);
      const assistantMessage = {
        id: makeId(),
        role: "assistant",
        content: response.reply,
        recommendations: response.recommendations,
      };
      setMessages([...nextMessages, assistantMessage]);
    } catch (err) {
      console.error(err);
      setError("Sorry, I had trouble reaching the assistant. Try again in a moment.");
      setMessages(nextMessages);
    } finally {
      setLoading(false);
    }
  };

  const placeholder = useMemo(
    () => "e.g. Cozy wooden coffee tables for a rustic living room",
    []
  );

  return (
    <section className="flex flex-1 flex-col gap-6">
      <div className="flex flex-1 flex-col gap-4 overflow-y-auto rounded-2xl bg-slate-900/60 p-6 shadow-lg">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {loading ? (
          <div className="flex justify-start text-sm text-slate-400">
            <span className="animate-pulse">Thinking…</span>
          </div>
        ) : null}
      </div>
      {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-lg"
      >
        <label htmlFor="message" className="text-sm font-medium text-slate-300">
          Describe what you need
        </label>
        <textarea
          id="message"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={placeholder}
          rows={3}
          className="w-full resize-none rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-sm text-slate-100 focus:border-primary-light focus:outline-none focus:ring-2 focus:ring-primary/40"
        />
        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-500">
            Powered by semantic search + generative copywriting
          </p>
          <button
            type="submit"
            disabled={loading}
            className="rounded-full bg-primary px-6 py-2 text-sm font-semibold text-white transition hover:bg-primary-light disabled:opacity-50"
          >
            {loading ? "Finding picks…" : "Send"}
          </button>
        </div>
      </form>
    </section>
  );
}

export default ChatPage;
