import { useState, useEffect } from "react";
import ChatBot from "./components/ChatBot";
import EvaluationPanel from "./components/EvaluationPanel";

function App() {
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [isScrolled, setIsScrolled] = useState(false);

  // Track scroll for button shadow
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="relative min-h-screen">
      {/* Blur overlay (conditionally rendered) */}
      {showEvaluation && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-30 backdrop-blur-sm z-30 animate-fadeIn"
          onClick={() => setShowEvaluation(false)}
        />
      )}

      {/* Chat Interface */}
      <div className={showEvaluation ? "filter blur-sm transition-all duration-300" : ""}>
        <ChatBot
          onNewSymbol={(symbol) => {
            setSelectedSymbol(symbol);
          }}
        />
      </div>

      {/* Enhanced Floating Action Button */}
      <button
        onClick={() => setShowEvaluation((prev) => !prev)}
        disabled={!selectedSymbol}
        className={`fixed bottom-6 right-6 z-50 flex items-center justify-center gap-2 ${
          selectedSymbol 
            ? "bg-gradient-to-br from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600" 
            : "bg-gray-400 cursor-not-allowed"
        } text-white px-5 py-3 rounded-full shadow-lg transition-all transform hover:scale-105 ${
          isScrolled ? "shadow-xl" : "shadow-md"
        } hover:scale-105`}
      >
        {showEvaluation ? (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            Close
          </>
        ) : (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
            </svg>
            Analyze {selectedSymbol || "Stock"}
          </>
        )}
      </button>

      {/* Evaluation Panel */}
      <div
        className={`fixed bottom-0 left-0 w-full z-40 transform transition-all duration-500 ease-in-out ${
          showEvaluation ? "translate-y-0" : "translate-y-full"
        }`}
      >
        {showEvaluation && (
          <EvaluationPanel
            symbol={selectedSymbol}
            onClose={() => setShowEvaluation(false)}
          />
        )}
      </div>
    </div>
  );
}

export default App;