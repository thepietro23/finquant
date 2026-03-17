import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Play, Shield, TrendingDown } from 'lucide-react';
import {
  ResponsiveContainer, XAxis, YAxis,
  CartesianGrid, LineChart, Line,
} from 'recharts';
import { api } from '../lib/api';
import type { ScenarioResult } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import Badge from '../components/ui/Badge';
import { staggerContainer } from '../lib/animations';

// Mock Monte Carlo paths
function genMonteCarloPaths(n = 50, days = 60) {
  const paths = [];
  for (let p = 0; p < n; p++) {
    let val = 100;
    const path = [];
    for (let d = 0; d < days; d++) {
      val *= (1 + (Math.random() - 0.48) * 0.04);
      path.push({ day: d, value: Math.round(val * 100) / 100 });
    }
    paths.push(path);
  }
  return paths;
}

export default function StressTesting() {
  const [loading, setLoading] = useState(false);
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [nStocks, setNStocks] = useState(10);
  const [nSim, setNSim] = useState(1000);
  const [mcPaths] = useState(() => genMonteCarloPaths());

  // Build chart data from first 5 paths for display
  const mcChartData = mcPaths[0].map((_, i) => {
    const point: Record<string, number> = { day: i };
    mcPaths.slice(0, 30).forEach((path, p) => {
      point[`p${p}`] = path[i].value;
    });
    return point;
  });

  async function runTest() {
    setLoading(true);
    try {
      const res = await api.stressTest(nStocks, nSim);
      setScenarios(res.scenarios);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Stress Testing"
        subtitle="VaR, CVaR, Monte Carlo simulation — 4 crash scenarios"
        icon={<AlertTriangle size={24} />}
      />

      {/* Controls */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">Stocks</label>
            <input type="number" value={nStocks} onChange={e => setNStocks(+e.target.value)}
              min={2} max={47}
              className="w-24 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">Simulations</label>
            <input type="number" value={nSim} onChange={e => setNSim(+e.target.value)}
              min={100} max={50000} step={100}
              className="w-28 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <button onClick={runTest} disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50">
            <Play size={16} />
            {loading ? 'Running...' : 'Generate Stress Test'}
          </button>
        </div>
      </Card>

      {/* Mock VaR Gauges */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <MetricCard title="VaR (95%)" value={-2.34} decimals={2} suffix="%" icon={<TrendingDown size={18} />} />
        <MetricCard title="CVaR (95%)" value={-3.87} decimals={2} suffix="%" icon={<Shield size={18} />} />
        <MetricCard title="Survival Rate" value={87.3} decimals={1} suffix="%" change={0.05} />
      </motion.div>

      {/* Monte Carlo Fan Chart */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Monte Carlo Simulation Paths
        </h2>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={mcChartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} label={{ value: 'Days', position: 'insideBottom', offset: -5 }} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              domain={['auto', 'auto']} />
            {mcPaths.slice(0, 30).map((_, i) => (
              <Line key={i} type="monotone" dataKey={`p${i}`} stroke="#C15F3C"
                strokeWidth={0.8} strokeOpacity={0.15} dot={false} animationDuration={0} />
            ))}
            {/* VaR line */}
            <Line type="monotone" dataKey="p0" stroke="#DC2626" strokeWidth={2}
              strokeDasharray="5 5" dot={false} name="Sample Path" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Scenario Results */}
      {scenarios.length > 0 && (
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Scenario Results</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2.5 font-medium text-text-secondary">Scenario</th>
                  <th className="text-right py-2.5 font-medium text-text-secondary">Mean Return</th>
                  <th className="text-right py-2.5 font-medium text-text-secondary">VaR 95%</th>
                  <th className="text-right py-2.5 font-medium text-text-secondary">CVaR 95%</th>
                  <th className="text-right py-2.5 font-medium text-text-secondary">Survival</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map(s => (
                  <tr key={s.scenario} className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-3 font-medium">
                      <Badge variant={s.scenario === 'normal' ? 'profit' : 'loss'}>
                        {s.scenario.replace(/_/g, ' ').toUpperCase()}
                      </Badge>
                    </td>
                    <td className="py-3 text-right font-mono">{s.mean_return}</td>
                    <td className="py-3 text-right font-mono text-loss">{s.var_95}</td>
                    <td className="py-3 text-right font-mono text-loss">{s.cvar_95}</td>
                    <td className="py-3 text-right font-mono">{s.survival_rate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
