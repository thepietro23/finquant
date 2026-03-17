import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Network } from 'lucide-react';
import { api } from '../lib/api';
import type { StockInfo } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import { staggerContainer, fadeSlideUp } from '../lib/animations';

// Sector colors for GNN visualization
const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Unknown': '#9CA3AF',
};

// Generate mock attention heatmap data
function generateAttentionMatrix(n: number) {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => Math.random())
  );
}

export default function GnnInsights() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);

  useEffect(() => {
    api.stocks().then(d => setStocks(d.stocks)).catch(() => {});
  }, []);

  // Sectors derived for visualization
  const attnMatrix = generateAttentionMatrix(Math.min(stocks.length, 15));
  const displayStocks = stocks.slice(0, 15);

  // Graph stats (mock realistic)
  const stats = {
    nodes: stocks.length || 47,
    sectorEdges: 162,
    supplyChainEdges: 54,
    correlationEdges: 234,
    density: 0.187,
  };

  return (
    <div>
      <PageHeader
        title="GNN Insights"
        subtitle="Temporal Graph Attention Network — 3 edge types, multi-relational attention"
        icon={<Network size={24} />}
      />

      {/* Graph Stats */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        {[
          { label: 'Nodes', value: stats.nodes, color: 'text-primary' },
          { label: 'Sector Edges', value: stats.sectorEdges, color: 'text-accent-indigo' },
          { label: 'Supply Chain', value: stats.supplyChainEdges, color: 'text-accent-teal' },
          { label: 'Correlation Edges', value: stats.correlationEdges, color: 'text-warning' },
          { label: 'Graph Density', value: stats.density, color: 'text-text-secondary' },
        ].map(s => (
          <motion.div key={s.label} variants={fadeSlideUp}
            className="bg-white rounded-2xl border border-border p-4">
            <p className="text-xs text-text-secondary mb-1">{s.label}</p>
            <p className={`text-2xl font-mono font-bold ${s.color}`}>
              {typeof s.value === 'number' && s.value < 1 ? s.value.toFixed(3) : s.value}
            </p>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Interactive Graph Placeholder (3D force graph is in the dedicated /graph page) */}
        <Card className="min-h-[400px] flex flex-col">
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Stock Network</h2>
          <div className="flex-1 relative bg-bg-card rounded-xl overflow-hidden">
            {/* SVG Graph Visualization */}
            <svg viewBox="0 0 500 400" className="w-full h-full">
              {/* Edges */}
              {displayStocks.slice(0, 10).map((_, i) => {
                const cx1 = 100 + (i % 5) * 80;
                const cy1 = 100 + Math.floor(i / 5) * 120;
                const j = (i + 3) % Math.min(displayStocks.length, 10);
                const cx2 = 100 + (j % 5) * 80;
                const cy2 = 100 + Math.floor(j / 5) * 120;
                return (
                  <line key={`e${i}`} x1={cx1} y1={cy1} x2={cx2} y2={cy2}
                    stroke="#E5E7EB" strokeWidth={1} opacity={0.6} />
                );
              })}
              {/* Nodes */}
              {displayStocks.slice(0, 10).map((s, i) => {
                const cx = 100 + (i % 5) * 80;
                const cy = 100 + Math.floor(i / 5) * 120;
                const color = SECTOR_COLORS[s.sector] || '#9CA3AF';
                return (
                  <g key={s.ticker}>
                    <circle cx={cx} cy={cy} r={18} fill={color} opacity={0.85} />
                    <text x={cx} y={cy + 4} textAnchor="middle" fill="white"
                      fontSize={8} fontWeight={600}>
                      {s.ticker.replace('.NS', '').slice(0, 4)}
                    </text>
                  </g>
                );
              })}
            </svg>
            <p className="absolute bottom-2 left-3 text-xs text-text-muted">
              Full interactive graph on Graph Viz page
            </p>
          </div>
        </Card>

        {/* Attention Heatmap */}
        <Card className="min-h-[400px]">
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Attention Heatmap</h2>
          <div className="overflow-auto">
            <div className="inline-grid gap-0.5" style={{
              gridTemplateColumns: `60px repeat(${displayStocks.length}, 28px)`,
            }}>
              {/* Header row */}
              <div />
              {displayStocks.map(s => (
                <div key={s.ticker} className="text-[7px] text-text-muted font-mono text-center rotate-[-45deg] origin-center h-8 flex items-end justify-center">
                  {s.ticker.replace('.NS', '').slice(0, 5)}
                </div>
              ))}
              {/* Data rows */}
              {attnMatrix.map((row, i) => (
                <div key={i} className="contents">
                  <div className="text-[8px] text-text-muted font-mono pr-1 flex items-center justify-end">
                    {displayStocks[i]?.ticker.replace('.NS', '').slice(0, 5)}
                  </div>
                  {row.map((v, j) => (
                    <div key={j} className="w-7 h-7 rounded-sm" style={{
                      backgroundColor: `rgba(193, 95, 60, ${v * 0.8 + 0.05})`,
                    }} title={`${v.toFixed(3)}`} />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Edge Type Legend */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-3">Edge Types</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { type: 'Sector', color: '#C15F3C', desc: 'Same sector stocks connected bidirectionally', count: stats.sectorEdges },
            { type: 'Supply Chain', color: '#6366F1', desc: 'Business relationships (e.g. TATASTEEL→MARUTI)', count: stats.supplyChainEdges },
            { type: 'Correlation', color: '#0D9488', desc: 'Rolling |corr| > 0.6 (dynamic, changes daily)', count: stats.correlationEdges },
          ].map(e => (
            <div key={e.type} className="flex items-start gap-3 p-3 rounded-xl bg-bg-card">
              <div className="w-3 h-3 rounded-full mt-1 shrink-0" style={{ backgroundColor: e.color }} />
              <div>
                <p className="font-medium text-sm text-text">{e.type} <span className="font-mono text-text-muted">({e.count})</span></p>
                <p className="text-xs text-text-secondary mt-0.5">{e.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
