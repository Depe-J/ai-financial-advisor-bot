import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// tooltip definitions for each metric card
// these appear when the user hovers over a metric so they know what it means
const METRIC_TOOLTIPS = {
  "Bootstrap N": "Number of resamples used to calculate confidence intervals. Higher = more reliable estimates.",
  "Initial Cash": "Starting portfolio value used in the backtest simulation.",
  "Final Value": "Portfolio value at the end of the backtest period after all trades.",
  "Losing Trades": "Number of closed trades that resulted in a loss compared to the buy price.",
  "Max Drawdown (%)": "Largest peak-to-trough drop in portfolio value. Shown as a negative number — closer to 0 is better.",
  "Profitable Trades": "Number of closed trades that resulted in a profit compared to the buy price.",
  "Return (%)": "Total percentage gain or loss over the backtest period compared to the starting cash.",
  "Return 95% CI": "Range within which the true return likely falls, based on 1000 bootstrap resamples. Wider range = more uncertainty.",
  "Sharpe 95% CI": "Range within which the true Sharpe ratio likely falls, based on 1000 bootstrap resamples.",
  "Sharpe Ratio": "Risk-adjusted return. Measures how much return you get per unit of risk. Above 1.0 is generally considered good.",
  "Total Trades": "Total number of completed trades executed during the backtest.",
  "Win Rate (%)": "Percentage of closed trades that were profitable. 100% means every trade made money.",
};

// small tooltip wrapper component - shows a dark bubble above the card on hover
const MetricCard = ({ metricKey, value, isProfit, isPositive, isNegative }) => {
  const tooltip = METRIC_TOOLTIPS[metricKey];

  return (
    <div className="relative group">
      <div
        className={`bg-white rounded-xl p-4 shadow-xs border ${
          isProfit
            ? isPositive ? "border-green-100 bg-green-50"
            : isNegative ? "border-red-100 bg-red-50"
            : "border-gray-100"
            : "border-gray-100"
        } transition-transform hover:scale-[1.02] cursor-default`}
      >
        <div className="flex justify-between items-start">
          <div>
            <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">{metricKey}</div>
            <div className={`text-xl font-bold ${
              isProfit
                ? isPositive ? "text-green-600"
                : isNegative ? "text-red-600"
                : "text-gray-800"
                : "text-gray-800"
            }`}>
              {String(value)}
            </div>
          </div>
        </div>
      </div>

      {/* tooltip bubble - only renders if we have a definition for this metric */}
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-gray-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 text-center leading-relaxed">
          {tooltip}
          {/* small arrow pointing down to the card */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
        </div>
      )}
    </div>
  );
};

const EvaluationPanel = ({ symbol: initialSymbol, onClose }) => {
  const [inputSymbol, setInputSymbol] = useState(initialSymbol || "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initialSymbol) {
      setInputSymbol(initialSymbol);
    }
  }, [initialSymbol]);

  const handleEvaluate = async () => {
    if (!inputSymbol.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://localhost:5050/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: inputSymbol.toUpperCase() }),
      });
  
      const data = await res.json();
  
      if (!res.ok || !data.summary || !Array.isArray(data.trades) || !Array.isArray(data.equity)) {
        throw new Error(data.message || "Invalid response from server.");
      }
  
      const equityWithDate = data.equity.map(item => ({
        ...item,
        date: new Date(item.date).toLocaleDateString()
      }));
  
      setResult({ ...data, equity: equityWithDate });
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50 flex items-end justify-center">
      <div className="w-full max-w-6xl bg-gray-50 rounded-t-3xl shadow-xl border-t border-gray-300 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-gray-900 to-gray-800 px-6 py-5 rounded-t-3xl flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-2 bg-blue-500 rounded-lg">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Strategy Performance Analytics</h2>
              <p className="text-gray-300 text-sm">Detailed analysis of trading strategy</p>
            </div>
          </div>
          <button 
            onClick={onClose} 
            className="text-gray-300 hover:text-white transition-colors p-1 rounded-full hover:bg-gray-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {/* Input Section */}
          <div className="bg-white rounded-xl p-5 mb-6 shadow-md border border-gray-200">
            <div className="flex flex-col sm:flex-row gap-3 items-center">
              <div className="relative flex-grow">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <input
                  value={inputSymbol}
                  onChange={(e) => setInputSymbol(e.target.value)}
                  placeholder="Enter stock symbol (e.g. AAPL, MSFT)"
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  onKeyDown={(e) => e.key === "Enter" && handleEvaluate()}
                />
              </div>
              <button
                onClick={handleEvaluate}
                disabled={loading}
                className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white px-6 py-3 rounded-lg w-full sm:w-auto font-medium disabled:opacity-70 transition-all flex items-center justify-center min-w-[150px]"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  "Run Analysis"
                )}
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-lg flex items-start">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-500 mr-3 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="text-sm font-medium text-red-800">Analysis Error</h3>
                <p className="text-sm text-red-600">{error}</p>
              </div>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-6">
              {/* Summary Cards - hover over any card to see what the metric means */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
                {Object.entries(result.summary || {})
                  .filter(([, value]) => value === null || ["string", "number", "boolean"].includes(typeof value))
                  .map(([key, value]) => {
                    const numVal = typeof value === "number"
                      ? value
                      : parseFloat(String(value).replace("%", ""));
                    const isProfit = key.toLowerCase().includes("profit") || key.toLowerCase().includes("return");
                    const isPositive = Number.isFinite(numVal) && numVal > 0;
                    const isNegative = Number.isFinite(numVal) && numVal < 0;
                    return (
                      <MetricCard
                        key={key}
                        metricKey={key}
                        value={value}
                        isProfit={isProfit}
                        isPositive={isPositive}
                        isNegative={isNegative}
                      />
                    );
                  })}
              </div>

              {/* Strategy Comparison Table */}
              {Array.isArray(result.summary["Strategy Comparison"]) &&
               result.summary["Strategy Comparison"].length > 0 && (
                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Strategy Comparison</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead>
                        <tr className="border-b border-gray-200">
                          {Object.keys(result.summary["Strategy Comparison"][0]).map(col => (
                            <th key={col} className="pb-2 pr-6 text-xs font-medium text-gray-500 uppercase tracking-wider">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.summary["Strategy Comparison"].map((row, i) => (
                          <tr key={i} className={`border-b border-gray-100 ${i === 2 ? "font-semibold bg-blue-50" : ""}`}>
                            {Object.values(row).map((val, j) => (
                              <td key={j} className="py-3 pr-6 text-gray-800">{val}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Charts Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Equity Chart */}
                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                      <span className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                        </svg>
                      </span>
                      Equity Curve
                    </h3>
                    <div className="text-sm text-gray-500">USD</div>
                  </div>
                  <div className="w-full h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.equity}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 11, fill: "#6b7280" }} 
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis 
                          tick={{ fontSize: 11, fill: "#6b7280" }} 
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: '#ffffff',
                            border: '1px solid #e5e7eb',
                            borderRadius: '0.5rem',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="equity" 
                          stroke="#3b82f6" 
                          strokeWidth={2} 
                          dot={false} 
                          activeDot={{ r: 6, strokeWidth: 2 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Price Trend */}
                <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                      <span className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mr-3">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </span>
                      Price Trend
                    </h3>
                    <div className="text-sm text-gray-500">USD</div>
                  </div>
                  <div className="w-full h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.equity}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 11, fill: "#6b7280" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis 
                          tick={{ fontSize: 11, fill: "#6b7280" }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: '#ffffff',
                            border: '1px solid #e5e7eb',
                            borderRadius: '0.5rem',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="close" 
                          stroke="#10b981" 
                          strokeWidth={2} 
                          dot={false} 
                          activeDot={{ r: 6, strokeWidth: 2 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EvaluationPanel;
