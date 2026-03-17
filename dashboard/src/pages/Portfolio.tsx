import { useEffect, useState } from 'react';
import { PieChart } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { StockInfo, MetricsResponse } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import MetricCard from '../components/ui/MetricCard';
import { staggerContainer } from '../lib/animations';
import { motion } from 'framer-motion';

export default function Portfolio() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    api.stocks().then(d => setStocks(d.stocks)).catch(() => {});
    // Generate mock returns and compute metrics via API
    const returns = Array.from({ length: 252 }, () => (Math.random() - 0.46) * 0.04);
    api.metrics(returns).then(setMetrics).catch(() => {});
  }, []);

  // Portfolio allocation by sector
  const sectorAlloc = stocks.reduce<Record<string, number>>((acc, s) => {
    acc[s.sector] = (acc[s.sector] || 0) + 1;
    return acc;
  }, {});
  const allocData = Object.entries(sectorAlloc)
    .sort((a, b) => b[1] - a[1])
    .map(([sector, count]) => ({
      sector, weight: Math.round((count / Math.max(stocks.length, 1)) * 100),
    }));

  return (
    <div>
      <PageHeader
        title="Portfolio Analysis"
        subtitle="Detailed portfolio metrics, sector allocation, and stock-level breakdown"
        icon={<PieChart size={24} />}
      />

      {metrics && (
        <motion.div variants={staggerContainer} initial="hidden" animate="visible"
          className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          <MetricCard title="Sharpe Ratio" value={metrics.sharpe_ratio} decimals={4} />
          <MetricCard title="Sortino Ratio" value={metrics.sortino_ratio} decimals={4} />
          <MetricCard title="Annual Return" value={metrics.annualized_return * 100} decimals={2} suffix="%" />
          <MetricCard title="Volatility" value={metrics.annualized_volatility * 100} decimals={2} suffix="%" />
          <MetricCard title="Max Drawdown" value={metrics.max_drawdown * 100} decimals={2} suffix="%" />
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Sector Allocation */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Sector Weights</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <BarChart data={allocData} layout="vertical" margin={{ top: 5, right: 10, bottom: 5, left: 80 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12, fill: '#9CA3AF' }}
                axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                tickFormatter={(v: number) => `${v}%`} />
              <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: '#6B7280' }}
                axisLine={false} tickLine={false} width={75} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                formatter={(v) => `${v}%`} />
              <Bar dataKey="weight" radius={[0, 6, 6, 0]} animationDuration={800}>
                {allocData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#C15F3C' : i === 1 ? '#6366F1' : i === 2 ? '#0D9488' : '#F59E0B'} opacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Stock List */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">
            All Stocks ({stocks.length})
          </h2>
          <div className="max-h-[300px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-border">
                  <th className="text-left py-2 font-medium text-text-secondary">#</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Ticker</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Sector</th>
                </tr>
              </thead>
              <tbody>
                {stocks.map((s, i) => (
                  <tr key={s.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-2 text-text-muted font-mono">{i + 1}</td>
                    <td className="py-2 font-mono font-medium text-text">{s.ticker.replace('.NS', '')}</td>
                    <td className="py-2 text-text-secondary">{s.sector}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
