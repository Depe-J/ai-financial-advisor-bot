import React, { useState, useRef, useEffect } from "react";

const ChatBot = ({ onNewSymbol }) => {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      // updated to reflect natural language input support - you can type a company name or ticker
      text: "Ask me about any stock — type a ticker like AAPL or just say 'tell me about Tesla'. I'll give you a BUY, SELL or HOLD recommendation with full analysis.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const userMessage = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setIsTyping(true);
    setInput("");
  
    // send the full raw input to the backend - it handles ticker resolution
    const rawInput = userMessage.text;

    console.log("[DEBUG] Sending request for:", rawInput);
  
    try {
      const res = await fetch("http://localhost:5050/advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: rawInput }),
      });
  
      const contentType = res.headers.get("content-type") || "";

      if (!contentType.includes("application/json")) {
        throw new Error("Invalid response format — expected JSON.");
      }

      const data = await res.json();
      console.log("[DEBUG] Bot response:", data);

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: data.message || `I wasn't able to retrieve data for that. Please try a ticker symbol like AAPL or TSLA.`,
          },
        ]);
      } else {
        // extract symbol from response for the evaluate button
        const symbolMatch = data.message?.match(/Recommendation:\s+\w+\s+(\w+)/);
        if (symbolMatch && onNewSymbol) onNewSymbol(symbolMatch[1]);

        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: data.message || "Something went wrong with the response.",
          },
        ]);
      }
    } catch (error) {
      console.error("[ERROR] Chat fetch failed:", error);

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Something went wrong. Please try again in a moment.",
        },
      ]);
    }
  
    setIsTyping(false);
    setLoading(false);
  };  

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 px-4">
      
      <div className="w-full max-w-2xl bg-white shadow-lg rounded-xl overflow-hidden flex flex-col h-[85vh] border border-gray-200 relative">
        
        <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white py-5 px-6 text-xl font-semibold flex items-center">
          <div className="w-3 h-3 bg-green-400 rounded-full mr-2 animate-pulse"></div>
          <span className="ml-6">Financial Advisor</span>
          <span className="ml-auto text-sm font-normal opacity-80">AI-powered</span>
        </div>

        
        <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
          <div className="space-y-3">
            {messages.map((msg, i) => (
              <div 
                key={i} 
                className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                    msg.sender === "user"
                      ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-none shadow-md"
                      : "bg-white text-gray-800 rounded-bl-none shadow-sm border border-gray-100"
                  } whitespace-pre-line transition-all duration-200`}
                >
                  {msg.text}
                </div>
              </div>
            ))}

            
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-white text-gray-500 text-xs px-4 py-2 rounded-2xl rounded-bl-none shadow border border-gray-100 flex items-center space-x-1">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span>Analyzing...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="flex items-center space-x-2 bg-gray-100 rounded-xl px-3 py-2 focus-within:ring-2 focus-within:ring-blue-400 focus-within:bg-white transition-all">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}

              placeholder="Ask about any stock (e.g. tell me about Tesla)"
              className="flex-1 bg-transparent border-none outline-none text-gray-700 placeholder-gray-400 text-sm py-2 px-2"
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className={`p-2 rounded-full ${
                loading 
                  ? "text-gray-400" 
                  : "text-white bg-blue-600 hover:bg-blue-700"
              } transition-all`}
            >
              {loading ? (
                <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clipRule="evenodd" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            {/* updated footer to reflect natural language input support */}
            Supports US-listed equities. Try natural language or ticker symbols.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;
