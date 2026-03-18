import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, Send, TrendingUp, TrendingDown, Minus,
  Newspaper, PieChart, ArrowUpRight, ArrowDownRight, RefreshCw,
  BarChart3, Briefcase,
} from 'lucide-react';
import { api } from '../lib/api';
import type {
  SentimentResponse, NewsSentimentResponse, NewsItem,
} from '../lib/api';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, PieChart as RPieChart, Pie,
} from 'recharts';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer, fadeSlideUp, scaleIn } from '../lib/animations';

const PAGE_INFO = {
  title: 'Sentiment Monitor — What Does This Page Show?',
  sections: [
    { heading: 'What is this page?', text: 'Uses FinBERT (a BERT model fine-tuned on financial text) to analyze real-time financial news from Google News RSS. Score ranges from -1 (very negative) to +1 (very positive).' },
    { heading: 'Live news feed', text: 'Real headlines fetched from Google News for 20 key NIFTY 50 stocks + market-wide queries. Each headline is analyzed by FinBERT in real-time — no mock data.' },
    { heading: 'Portfolio impact', text: 'Sentiment-adjusted weights: positive news → higher allocation, negative → lower. Shows how news could shift your portfolio vs equal-weight baseline.' },
    { heading: 'Sector sentiment', text: 'Aggregated sentiment per sector from multiple headlines. Helps identify which sectors have positive/negative news momentum right now.' },
    { heading: 'Why FinBERT?', text: 'General NLP models misinterpret financial language ("bearish" is negative in finance, neutral elsewhere). FinBERT was trained on 10,000+ financial texts for domain-specific accuracy.' },
    { heading: 'How is it used?', text: 'Sentiment scores are fed as input features to the RL agent. The agent considers market mood (from news) alongside price data when making allocation decisions.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Headlines Analyzed': {
    what: 'Total number of real financial news headlines fetched and analyzed by FinBERT.',
    why: 'More headlines = more robust sentiment signal. A single headline can be misleading; aggregating many gives a clearer picture.',
    how: 'Google News RSS fetched for 20 key stocks + 2 market-wide queries. Each headline passed through FinBERT for sentiment scoring.',
    good: '50-100 headlines is ideal for NIFTY 50 coverage. <20 = too few for reliable sector-level signals.',
  },
  'Market Mood': {
    what: 'Overall market sentiment derived from averaging all headline scores. Bullish (>0.08), Bearish (<-0.08), or Neutral.',
    why: 'A quick pulse check on the market. When most news is positive, market tends to be in risk-on mode; negative = risk-off.',
    how: 'Simple average of all FinBERT scores. Threshold: >0.08 = Bullish, <-0.08 = Bearish, else Neutral.',
    good: 'Score between -0.1 and +0.1 = typical market. Beyond ±0.2 = strong sentiment (unusual, often around earnings season or macro events).',
  },
  'Avg Score': {
    what: 'Mean sentiment score across all analyzed headlines. Ranges from -1.0 (extremely negative) to +1.0 (extremely positive).',
    why: 'Quantifies overall market sentiment into a single number. Used as an input feature for the RL agent.',
    how: 'Average of FinBERT scores: score = positive_prob - negative_prob. Each headline contributes equally.',
    good: 'Most days score between -0.05 and +0.15 (slight positive bias in financial news). Extreme values (>0.3 or <-0.3) are rare.',
  },
  'Top Mover': {
    what: 'The stock with the largest sentiment-driven weight change from equal-weight baseline.',
    why: 'Identifies which stock sentiment would most affect portfolio allocation. High positive = overweight, high negative = underweight.',
    how: 'Weight adjustment = base_weight × (1 + sentiment × 2.0), then normalized. Biggest absolute change = top mover.',
    good: 'Weight changes of ±0.5% are normal. >±1% = strong sentiment signal for that stock.',
  },
};

const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Infra': '#14B8A6', 'Others': '#9CA3AF',
  'Market': '#374151', 'Unknown': '#9CA3AF',
};

function ScoreBar({ score }: { score: number }) {
  const pct = ((score + 1) / 2) * 100;
  return (
    <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="h-full rounded-full"
        style={{
          background: score >= 0.1
            ? 'linear-gradient(90deg, #F59E0B, #16A34A)'
            : score <= -0.1
            ? 'linear-gradient(90deg, #DC2626, #F59E0B)'
            : 'linear-gradient(90deg, #F59E0B 40%, #EAB308 60%)',
        }}
      />
    </div>
  );
}

function NewsCard({ item, index }: { item: NewsItem; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
      className="group p-3 rounded-xl bg-white border border-border-light hover:border-primary/30
        hover:shadow-[0_4px_16px_rgba(193,95,60,0.08)] transition-all duration-300"
    >
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-white text-xs font-bold
          ${item.label === 'positive' ? 'bg-profit' : item.label === 'negative' ? 'bg-loss' : 'bg-gray-400'}`}>
          {item.score > 0 ? '+' : ''}{(item.score * 100).toFixed(0)}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text leading-snug line-clamp-2 group-hover:text-primary transition-colors">
            {item.headline}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            {item.ticker !== 'MARKET' && (
              <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-bg-card border border-border-light"
                style={{ borderLeftColor: SECTOR_COLORS[item.sector] || '#9CA3AF', borderLeftWidth: 2 }}>
                {item.ticker}
              </span>
            )}
            <span className="text-[10px] text-text-muted">{item.published}</span>
            {item.source && <span className="text-[10px] text-text-muted truncate max-w-[120px]">{item.source}</span>}
          </div>
        </div>
        <Badge variant={item.label === 'positive' ? 'profit' : item.label === 'negative' ? 'loss' : 'neutral'}>
          {item.label === 'positive' ? <TrendingUp size={10} /> : item.label === 'negative' ? <TrendingDown size={10} /> : <Minus size={10} />}
          <span className="ml-1">{item.label}</span>
        </Badge>
      </div>
    </motion.div>
  );
}

export default function Sentiment() {
  // Manual analysis state
  const [text, setText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SentimentResponse | null>(null);

  // Live news state
  const [newsData, setNewsData] = useState<NewsSentimentResponse | null>(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsError, setNewsError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'news' | 'portfolio' | 'sectors'>('news');

  function loadNewsSentiment() {
    setNewsLoading(true);
    setNewsError(null);
    api.newsSentiment()
      .then(d => { setNewsData(d); setNewsLoading(false); })
      .catch(e => {
        setNewsError(e instanceof Error ? e.message : 'Failed to fetch news sentiment');
        setNewsLoading(false);
      });
  }

  useEffect(() => { loadNewsSentiment(); }, []);

  async function analyze() {
    if (!text.trim()) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await api.sentiment(text);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Sentiment analysis failed — is the backend running?');
    } finally {
      setAnalyzing(false);
    }
  }

  // Derived data
  const topMover = newsData?.portfolio_impact[0];
  const moodValue = newsData ? (newsData.market_mood === 'Bullish' ? 1 : newsData.market_mood === 'Bearish' ? -1 : 0) : 0;

  // Score distribution for pie chart
  const distData = newsData ? Object.entries(newsData.score_distribution)
    .map(([key, value]) => ({
      name: key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value,
      color: key === 'very_positive' ? '#16A34A' : key === 'positive' ? '#86EFAC'
        : key === 'neutral' ? '#F59E0B' : key === 'negative' ? '#FCA5A5' : '#DC2626',
    }))
    .filter(d => d.value > 0) : [];

  // Sector bar chart data
  const sectorChartData = newsData?.sector_sentiment.map(s => ({
    sector: s.sector.length > 10 ? s.sector.slice(0, 10) : s.sector,
    score: s.avg_score,
    headlines: s.n_headlines,
  })) || [];

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Sentiment Monitor"
          subtitle="FinBERT NLP — real-time news sentiment from Google News"
          icon={<MessageSquare size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Metric Cards */}
      {newsData && (
        <>
          <motion.div variants={staggerContainer} initial="hidden" animate="visible"
            className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
            <MetricCard title="Headlines Analyzed" value={newsData.n_headlines} decimals={0}
              icon={<Newspaper size={18} />}
              onClick={() => setExpandedMetric(m => m === 'Headlines Analyzed' ? null : 'Headlines Analyzed')}
              active={expandedMetric === 'Headlines Analyzed'} />
            <MetricCard title="Market Mood" value={moodValue} decimals={0}
              prefix={newsData.market_mood === 'Bullish' ? '↑ ' : newsData.market_mood === 'Bearish' ? '↓ ' : '→ '}
              suffix={` ${newsData.market_mood}`}
              icon={newsData.market_mood === 'Bullish' ? <TrendingUp size={18} /> : newsData.market_mood === 'Bearish' ? <TrendingDown size={18} /> : <Minus size={18} />}
              onClick={() => setExpandedMetric(m => m === 'Market Mood' ? null : 'Market Mood')}
              active={expandedMetric === 'Market Mood'} />
            <MetricCard title="Avg Score" value={newsData.avg_score} decimals={4}
              icon={<BarChart3 size={18} />}
              onClick={() => setExpandedMetric(m => m === 'Avg Score' ? null : 'Avg Score')}
              active={expandedMetric === 'Avg Score'} />
            <MetricCard title="Top Mover"
              value={topMover ? topMover.weight_change : 0} decimals={2} suffix="%"
              prefix={topMover ? `${topMover.ticker} ` : ''}
              icon={<Briefcase size={18} />}
              onClick={() => setExpandedMetric(m => m === 'Top Mover' ? null : 'Top Mover')}
              active={expandedMetric === 'Top Mover'} />
          </motion.div>
          <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />
        </>
      )}

      {/* Manual Analysis */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-3">Analyze Custom Text</h2>
        <div className="flex gap-3">
          <input
            type="text" value={text} onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && analyze()}
            placeholder="Enter any financial headline..."
            maxLength={500}
            className="flex-1 px-4 py-2.5 border border-border rounded-xl text-sm focus:outline-none focus:border-primary transition-colors"
          />
          <button onClick={analyze} disabled={analyzing || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50">
            <Send size={16} />
            {analyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {error && <p className="mt-3 text-sm text-loss">{error}</p>}

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-bg-card rounded-xl overflow-hidden"
            >
              <div className="flex items-center gap-3 mb-3">
                <Badge variant={result.label === 'positive' ? 'profit' : result.label === 'negative' ? 'loss' : 'neutral'}>
                  {result.label === 'positive' ? <TrendingUp size={12} /> : result.label === 'negative' ? <TrendingDown size={12} /> : <Minus size={12} />}
                  <span className="ml-1">{result.label.toUpperCase()}</span>
                </Badge>
                <span className="font-mono text-xl font-bold text-text">{result.score.toFixed(4)}</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Positive</p>
                  <p className="font-mono font-semibold text-profit">{(result.positive * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Negative</p>
                  <p className="font-mono font-semibold text-loss">{(result.negative * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Neutral</p>
                  <p className="font-mono font-semibold text-text-secondary">{(result.neutral * 100).toFixed(1)}%</p>
                </div>
              </div>
              <ScoreBar score={result.score} />
              <div className="flex justify-between text-[10px] text-text-muted mt-1">
                <span>-1.0 (Very Negative)</span><span>0 (Neutral)</span><span>+1.0 (Very Positive)</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Live News Section */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="font-display font-bold text-lg text-secondary">Live News Feed</h2>
            {newsData && (
              <span className="text-xs text-text-muted bg-bg-card px-2 py-0.5 rounded-md border border-border-light">
                {newsData.n_headlines} headlines
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Tab switcher */}
            <div className="flex bg-bg-card rounded-lg border border-border-light p-0.5">
              {(['news', 'portfolio', 'sectors'] as const).map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === tab
                      ? 'bg-primary text-white shadow-sm'
                      : 'text-text-secondary hover:text-text'
                  }`}>
                  {tab === 'news' ? 'News' : tab === 'portfolio' ? 'Portfolio Impact' : 'Sectors'}
                </button>
              ))}
            </div>
            <button onClick={loadNewsSentiment} disabled={newsLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary bg-primary-subtle rounded-lg hover:bg-primary-light transition-colors disabled:opacity-50">
              <RefreshCw size={12} className={newsLoading ? 'animate-spin' : ''} />
              {newsLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>

        {newsError && <p className="text-sm text-loss mb-3">{newsError}</p>}

        {newsLoading && !newsData && (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary mb-4" />
            <p className="text-sm text-text-secondary">Fetching live news & running FinBERT analysis...</p>
            <p className="text-xs text-text-muted mt-1">This may take 15-30 seconds (fetching 20+ stocks from Google News)</p>
          </div>
        )}

        {newsData && (
          <AnimatePresence mode="wait">
            {/* NEWS TAB */}
            {activeTab === 'news' && (
              <motion.div key="news" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
                {newsData.news.map((item, i) => (
                  <NewsCard key={i} item={item} index={i} />
                ))}
              </motion.div>
            )}

            {/* PORTFOLIO IMPACT TAB */}
            {activeTab === 'portfolio' && (
              <motion.div key="portfolio" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <p className="text-xs text-text-secondary mb-3">
                  How sentiment would adjust equal-weight portfolio. Positive sentiment → overweight, negative → underweight.
                </p>
                <div className="overflow-auto max-h-[480px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-white z-10">
                      <tr className="border-b border-border">
                        <th className="text-left py-2 font-medium text-text-secondary">Stock</th>
                        <th className="text-left py-2 font-medium text-text-secondary">Sector</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Sentiment</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Base Wt</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Adj Wt</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {newsData.portfolio_impact.slice(0, 25).map((h, i) => (
                        <motion.tr key={h.ticker}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.03 }}
                          className="border-b border-border-light hover:bg-bg-card transition-colors">
                          <td className="py-2 font-mono font-medium">{h.ticker}</td>
                          <td className="py-2">
                            <span className="text-xs px-2 py-0.5 rounded-full bg-bg-card"
                              style={{ color: SECTOR_COLORS[h.sector] || '#9CA3AF' }}>
                              {h.sector}
                            </span>
                          </td>
                          <td className="py-2 text-right">
                            <span className={`font-mono text-xs font-semibold ${
                              h.sentiment_score > 0.05 ? 'text-profit' : h.sentiment_score < -0.05 ? 'text-loss' : 'text-text-muted'
                            }`}>
                              {h.sentiment_score > 0 ? '+' : ''}{h.sentiment_score.toFixed(3)}
                            </span>
                          </td>
                          <td className="py-2 text-right font-mono text-text-muted">{h.base_weight.toFixed(1)}%</td>
                          <td className="py-2 text-right font-mono font-semibold">{h.adjusted_weight.toFixed(2)}%</td>
                          <td className="py-2 text-right">
                            <span className={`inline-flex items-center gap-0.5 font-mono text-xs font-semibold ${
                              h.weight_change > 0 ? 'text-profit' : h.weight_change < 0 ? 'text-loss' : 'text-text-muted'
                            }`}>
                              {h.weight_change > 0 ? <ArrowUpRight size={10} /> : h.weight_change < 0 ? <ArrowDownRight size={10} /> : null}
                              {h.weight_change > 0 ? '+' : ''}{h.weight_change.toFixed(2)}%
                            </span>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {/* SECTORS TAB */}
            {activeTab === 'sectors' && (
              <motion.div key="sectors" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Sector bar chart */}
                  <div>
                    <h3 className="text-sm font-medium text-text-secondary mb-3">Sector Sentiment Scores</h3>
                    <ResponsiveContainer width="100%" height={300} minHeight={1}>
                      <BarChart data={sectorChartData} layout="vertical" margin={{ top: 5, right: 10, bottom: 5, left: 70 }}>
                        <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 11, fill: '#9CA3AF' }}
                          axisLine={{ stroke: '#E5E7EB' }} tickLine={false} domain={[-0.5, 0.5]} />
                        <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: '#6B7280' }}
                          axisLine={false} tickLine={false} width={65} />
                        <Tooltip
                          contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12, fontSize: 12 }}
                          formatter={(v: number) => [v.toFixed(4), 'Score']}
                        />
                        <Bar dataKey="score" name="Sentiment" radius={[0, 6, 6, 0]} animationDuration={1000}>
                          {sectorChartData.map((s, i) => (
                            <Cell key={i} fill={s.score >= 0.05 ? '#16A34A' : s.score <= -0.05 ? '#DC2626' : '#F59E0B'} opacity={0.8} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Sector detail cards */}
                  <div className="space-y-2 max-h-[340px] overflow-y-auto">
                    {newsData.sector_sentiment.map((s, i) => (
                      <motion.div key={s.sector}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06 }}
                        className="p-3 rounded-xl bg-bg-card border border-border-light"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: SECTOR_COLORS[s.sector] || '#9CA3AF' }} />
                            <span className="font-medium text-sm">{s.sector}</span>
                            <span className="text-[10px] text-text-muted">({s.n_headlines} headlines)</span>
                          </div>
                          <span className={`font-mono text-sm font-bold ${
                            s.avg_score > 0.05 ? 'text-profit' : s.avg_score < -0.05 ? 'text-loss' : 'text-text-muted'
                          }`}>
                            {s.avg_score > 0 ? '+' : ''}{s.avg_score.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden flex">
                            <div className="h-full bg-profit rounded-l-full" style={{ width: `${s.positive_pct}%` }} />
                            <div className="h-full bg-amber-300" style={{ width: `${100 - s.positive_pct - s.negative_pct}%` }} />
                            <div className="h-full bg-loss rounded-r-full" style={{ width: `${s.negative_pct}%` }} />
                          </div>
                        </div>
                        <div className="flex justify-between text-[10px] text-text-muted mt-1">
                          <span className="text-profit">{s.positive_pct}% positive</span>
                          <span className="text-loss">{s.negative_pct}% negative</span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </Card>

      {/* Score Distribution */}
      {newsData && distData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <h2 className="font-display font-bold text-lg text-secondary mb-3">Score Distribution</h2>
            <p className="text-xs text-text-secondary mb-3">
              How headlines are distributed across sentiment buckets. A healthy market has mostly neutral headlines.
            </p>
            <ResponsiveContainer width="100%" height={240} minHeight={1}>
              <RPieChart>
                <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={55} outerRadius={95} paddingAngle={3}
                  animationBegin={200} animationDuration={800}>
                  {distData.map((d, i) => (
                    <Cell key={i} fill={d.color} stroke="#fff" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #E5E7EB', fontSize: 12 }}
                  formatter={(v: number) => [`${v} headlines`, 'Count']} />
              </RPieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap justify-center gap-3 mt-2">
              {distData.map(d => (
                <span key={d.name} className="flex items-center gap-1.5 text-[10px] text-text-secondary">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: d.color }} />
                  {d.name} ({d.value})
                </span>
              ))}
            </div>
          </Card>

          {/* How Sentiment Affects Portfolio */}
          <Card>
            <h2 className="font-display font-bold text-lg text-secondary mb-3">How Sentiment Adjusts Portfolio</h2>
            <div className="space-y-4 text-sm">
              <div className="p-3 rounded-xl bg-profit-light/50 border border-profit/20">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp size={14} className="text-profit" />
                  <span className="font-semibold text-profit">Positive Sentiment → Overweight</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Stocks with positive news get higher allocation. Formula: weight × (1 + score × 2.0). A stock with +0.3 sentiment gets ~60% more weight.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-loss-light/50 border border-loss/20">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingDown size={14} className="text-loss" />
                  <span className="font-semibold text-loss">Negative Sentiment → Underweight</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Stocks with negative news get lower allocation. A stock with -0.3 sentiment gets ~40% less weight. This acts as risk management.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-amber-50 border border-amber-200/50">
                <div className="flex items-center gap-2 mb-1">
                  <Minus size={14} className="text-amber-600" />
                  <span className="font-semibold text-amber-700">Neutral → No Change</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Stocks with neutral news keep their equal-weight baseline. Most stocks fall here — only strong signals move weights significantly.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-bg-card border border-border-light">
                <div className="flex items-center gap-2 mb-1">
                  <PieChart size={14} className="text-text-secondary" />
                  <span className="font-semibold text-text-secondary">No News → Sector Average</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Stocks without specific news use their sector's average sentiment. If Banking sector is +0.15, all banking stocks without news inherit that.
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
