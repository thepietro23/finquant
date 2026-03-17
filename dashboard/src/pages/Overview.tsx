import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, TrendingUp, TrendingDown, BarChart3, Activity } from 'lucide-react';
import { staggerContainer } from '../lib/animations';
import { api } from '../lib/api';
import type { StockInfo, HealthResponse } from '../lib/api';
import MetricCard from '../components/ui/MetricCard';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import PerformanceChart from '../components/charts/PerformanceChart';
import SectorDonut from '../components/charts/SectorDonut';

// Generate realistic mock data for demo
function generatePerformanceData() {
  const data = [];
  let portfolio = 0, nifty = 0;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  for (let m = 0; m < 12; m++) {
    for (let d = 0; d < 20; d++) {
      portfolio += (Math.random() - 0.44) * 1.2;
      nifty += (Math.random() - 0.46) * 0.8;
      data.push({
        date: `${months[m]} ${d + 1}`,
        portfolio: Math.round(portfolio * 100) / 100,
        nifty: Math.round(nifty * 100) / 100,
      });
    }
  }
  return data;
}

function generateSparkData(trend: number) {
  return Array.from({ length: 30 }, (_, i) =>
    50 + trend * i + (Math.random() - 0.5) * 10
  );
}

export default function Overview() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const perfData = useState(() => generatePerformanceData())[0];

  useEffect(() => {
    api.stocks().then(d => setStocks(d.stocks)).catch(() => {});
    api.health().then(setHealth).catch(() => {});
  }, []);

  // Aggregate sectors
  const sectorCounts: Record<string, number> = {};
  stocks.forEach(s => { sectorCounts[s.sector] = (sectorCounts[s.sector] || 0) + 1; });
  const sectorData = Object.entries(sectorCounts).map(([name, count]) => ({
    name, value: Math.round((count / Math.max(stocks.length, 1)) * 100),
  }));

  // Top holdings mock (realistic NIFTY weights)
  const topHoldings = stocks.slice(0, 10).map((s, i) => ({
    ...s, weight: (20 - i * 1.8).toFixed(1),
    ret: ((Math.random() - 0.4) * 30).toFixed(1),
  }));

  return (
    <div>
      <PageHeader
        title="Portfolio Overview"
        subtitle={health ? `${health.project} — ${health.version} — ${health.phases_complete} phases deployed` : 'Loading...'}
        icon={<LayoutDashboard size={24} />}
      />

      {/* Metric Cards Row */}
      <motion.div
        variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
      >
        <MetricCard
          title="Portfolio Value" prefix="₹" value={12435678} decimals={0}
          suffix="" change={0.243} sparkData={generateSparkData(0.5)}
          icon={<TrendingUp size={18} />}
        />
        <MetricCard
          title="Sharpe Ratio" value={1.65} decimals={2}
          change={0.078} sparkData={generateSparkData(0.3)}
          icon={<BarChart3 size={18} />}
        />
        <MetricCard
          title="Max Drawdown" value={-12.3} decimals={1} suffix="%"
          change={-0.021} sparkData={generateSparkData(-0.2)}
          icon={<TrendingDown size={18} />}
        />
        <MetricCard
          title="Annual Return" value={24.3} decimals={1} suffix="%"
          change={0.034} sparkData={generateSparkData(0.4)}
          icon={<Activity size={18} />}
        />
      </motion.div>

      {/* Performance Chart */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display font-bold text-lg text-secondary">Performance vs NIFTY 50</h2>
          <div className="flex gap-1">
            {['1W', '1M', '3M', '6M', '1Y'].map(period => (
              <button key={period}
                className="px-3 py-1 text-xs font-medium rounded-lg transition-colors
                  hover:bg-primary-subtle hover:text-primary text-text-secondary"
              >
                {period}
              </button>
            ))}
          </div>
        </div>
        <PerformanceChart data={perfData} />
      </Card>

      {/* Two columns: Sector + Holdings */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Sector Allocation</h2>
          {sectorData.length > 0 ? (
            <SectorDonut data={sectorData} />
          ) : (
            <div className="h-60 flex items-center justify-center text-text-muted">Loading sectors...</div>
          )}
        </Card>

        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Top Holdings</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 font-medium text-text-secondary">Stock</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Sector</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Weight</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Return</th>
                </tr>
              </thead>
              <tbody>
                {topHoldings.map(h => (
                  <tr key={h.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-2.5 font-mono font-medium text-text">{h.ticker.replace('.NS', '')}</td>
                    <td className="py-2.5 text-text-secondary">{h.sector}</td>
                    <td className="py-2.5 text-right font-mono">{h.weight}%</td>
                    <td className={`py-2.5 text-right font-mono font-medium ${
                      Number(h.ret) >= 0 ? 'text-profit' : 'text-loss'
                    }`}>
                      {Number(h.ret) > 0 ? '+' : ''}{h.ret}%
                    </td>
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
