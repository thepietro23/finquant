import { useState, useEffect } from 'react';
import { MessageSquare, Send, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { api } from '../lib/api';
import type { SentimentResponse, StockInfo } from '../lib/api';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell,
} from 'recharts';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import Badge from '../components/ui/Badge';

// Sample financial headlines for demo
const SAMPLE_HEADLINES = [
  "Reliance Industries reports record quarterly profit of ₹19,000 Cr",
  "HDFC Bank faces regulatory scrutiny over lending practices",
  "Infosys wins $2 billion digital transformation deal",
  "Market crash fears as global recession concerns mount",
  "TCS announces 15% dividend, shares surge 3%",
  "NIFTY 50 hits all-time high driven by banking stocks",
  "Adani Group stocks fall sharply amid debt concerns",
  "RBI holds interest rates steady at 6.5%",
];

export default function Sentiment() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentResponse | null>(null);
  const [batchResults, setBatchResults] = useState<SentimentResponse[]>([]);
  const [stocks, setStocks] = useState<StockInfo[]>([]);

  useEffect(() => {
    api.stocks().then(d => setStocks(d.stocks)).catch(() => {});
  }, []);

  async function analyze() {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await api.sentiment(text);
      setResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  async function analyzeBatch() {
    setLoading(true);
    try {
      const res = await api.sentimentBatch(SAMPLE_HEADLINES);
      setBatchResults(res.results);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  const labelIcon = (label: string) => {
    if (label === 'positive') return <TrendingUp size={14} />;
    if (label === 'negative') return <TrendingDown size={14} />;
    return <Minus size={14} />;
  };

  // Sector sentiment mock using stocks
  const sectorSentiment = [...new Set(stocks.map(s => s.sector))].slice(0, 8).map(sector => ({
    sector: sector.length > 10 ? sector.slice(0, 10) : sector,
    score: Math.round((Math.random() - 0.4) * 0.8 * 100) / 100,
  }));

  return (
    <div>
      <PageHeader
        title="Sentiment Monitor"
        subtitle="FinBERT NLP — real-time sentiment analysis on financial text"
        icon={<MessageSquare size={24} />}
      />

      {/* Single Text Analysis */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">Analyze Text</h2>
        <div className="flex gap-3">
          <input
            type="text" value={text} onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && analyze()}
            placeholder="Enter financial news headline..."
            className="flex-1 px-4 py-2.5 border border-border rounded-xl text-sm focus:outline-none focus:border-primary"
          />
          <button onClick={analyze} disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50">
            <Send size={16} />
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {result && (
          <div className="mt-4 p-4 bg-bg-card rounded-xl">
            <div className="flex items-center gap-3 mb-3">
              <Badge variant={result.label === 'positive' ? 'profit' : result.label === 'negative' ? 'loss' : 'neutral'}>
                {labelIcon(result.label)} {result.label.toUpperCase()}
              </Badge>
              <span className="font-mono text-lg font-bold text-text">{result.score.toFixed(4)}</span>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <p className="text-text-muted text-xs mb-1">Positive</p>
                <p className="font-mono font-semibold text-profit">{(result.positive * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-text-muted text-xs mb-1">Negative</p>
                <p className="font-mono font-semibold text-loss">{(result.negative * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-text-muted text-xs mb-1">Neutral</p>
                <p className="font-mono font-semibold text-text-secondary">{(result.neutral * 100).toFixed(1)}%</p>
              </div>
            </div>
            {/* Score bar */}
            <div className="mt-3 h-3 bg-border-light rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-500" style={{
                width: `${(result.score + 1) / 2 * 100}%`,
                background: result.score >= 0
                  ? `linear-gradient(90deg, #F59E0B, #16A34A)`
                  : `linear-gradient(90deg, #DC2626, #F59E0B)`,
              }} />
            </div>
            <div className="flex justify-between text-[10px] text-text-muted mt-1">
              <span>-1.0 (Very Negative)</span>
              <span>0 (Neutral)</span>
              <span>+1.0 (Very Positive)</span>
            </div>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Batch Analysis */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display font-bold text-lg text-secondary">Batch Headlines</h2>
            <button onClick={analyzeBatch} disabled={loading}
              className="px-3 py-1.5 text-xs font-medium text-primary bg-primary-subtle rounded-lg hover:bg-primary-light transition-colors disabled:opacity-50">
              {loading ? 'Running...' : 'Analyze All'}
            </button>
          </div>
          <div className="space-y-2 max-h-[360px] overflow-y-auto">
            {(batchResults.length > 0 ? batchResults : SAMPLE_HEADLINES.map(t => ({
              text: t, score: 0, label: 'neutral', positive: 0, negative: 0, neutral: 1,
            }))).map((r, i) => (
              <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-bg-card hover:bg-primary-subtle/30 transition-colors">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-text truncate">{r.text}</p>
                </div>
                {r.score !== 0 && (
                  <Badge variant={r.label === 'positive' ? 'profit' : r.label === 'negative' ? 'loss' : 'neutral'}>
                    {r.score > 0 ? '+' : ''}{r.score.toFixed(2)}
                  </Badge>
                )}
              </div>
            ))}
          </div>
        </Card>

        {/* Sector Sentiment */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Sector Sentiment</h2>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={sectorSentiment} layout="vertical" margin={{ top: 5, right: 10, bottom: 5, left: 60 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12, fill: '#9CA3AF' }}
                axisLine={{ stroke: '#E5E7EB' }} tickLine={false} domain={[-0.5, 0.5]} />
              <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: '#6B7280' }}
                axisLine={false} tickLine={false} width={60} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                formatter={(v) => Number(v).toFixed(3)} />
              <Bar dataKey="score" name="Sentiment Score" radius={[0, 6, 6, 0]} animationDuration={800}>
                {sectorSentiment.map((s, i) => (
                  <Cell key={i} fill={s.score >= 0 ? '#16A34A' : '#DC2626'} opacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
