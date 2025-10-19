import RecommendationList from "./RecommendationList";

function ChatMessage({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-3xl rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-md ${
          isUser ? "bg-primary text-white" : "bg-slate-800 text-slate-100"
        }`}
      >
        <p>{message.content}</p>
        {!isUser && message.recommendations?.length ? (
          <div className="mt-4">
            <RecommendationList items={message.recommendations} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default ChatMessage;
