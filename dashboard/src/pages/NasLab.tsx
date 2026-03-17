import { useMemo } from 'react';
import { FlaskConical, Layers, Cpu, Zap } from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
} from 'recharts';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import { staggerContainer } from '../lib/animations';
import { motion } from 'framer-motion';

const OPS = ['Linear', 'Conv1D', 'Attention', 'Skip', 'None'];
const OP_COLORS = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B', '#D1D5DB'];

function genAlphaConvergence() {
  const data = [];
  const alphas = OPS.map(() => 0.2);
  for (let epoch = 0; epoch <= 30; epoch++) {
    const entry: Record<string, number> = { epoch };
    // Simulate convergence — one op dominates over time
    alphas[0] += (Math.random() * 0.03);
    alphas[2] += (Math.random() * 0.02);
    alphas[4] -= (Math.random() * 0.02);
    const sum = alphas.reduce((a, b) => a + b, 0);
    OPS.forEach((op, i) => { entry[op] = Math.round((alphas[i] / sum) * 1000) / 1000; });
    data.push(entry);
  }
  return data;
}

export default function NasLab() {
  const alphaData = useMemo(genAlphaConvergence, []);

  const compareData = [
    { name: 'NAS-Found', sharpe: 1.72, return: 26.1, drawdown: 10.2 },
    { name: 'Hand-Designed', sharpe: 1.52, return: 22.3, drawdown: 13.8 },
  ];

  const archOps = ['Linear', 'Attention', 'Linear', 'Skip']; // Best found arch

  return (
    <div>
      <PageHeader
        title="NAS Lab"
        subtitle="DARTS architecture search for T-GAT + RL policy grid search"
        icon={<FlaskConical size={24} />}
      />

      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard title="Search Epochs" value={30} decimals={0} icon={<Layers size={18} />} />
        <MetricCard title="Best Op" value={0} decimals={0} prefix="Linear"
          icon={<Cpu size={18} />} />
        <MetricCard title="NAS Sharpe" value={1.72} decimals={2} change={0.13} icon={<Zap size={18} />} />
        <MetricCard title="Improvement" value={13.2} decimals={1} suffix="%" change={0.132} />
      </motion.div>

      {/* Architecture Diagram */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">Best Architecture Found</h2>
        <div className="flex items-center gap-3 overflow-x-auto py-4">
          <div className="shrink-0 px-4 py-3 bg-bg-card rounded-xl border border-border text-sm font-mono text-text-secondary">
            Input (21 feat)
          </div>
          {archOps.map((op, i) => (
            <div key={i} className="flex items-center gap-3">
              <svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#D1D5DB" strokeWidth="2" /></svg>
              <div className={`shrink-0 px-4 py-3 rounded-xl border-2 text-sm font-semibold ${
                op === 'Linear' ? 'border-primary bg-primary-subtle text-primary' :
                op === 'Attention' ? 'border-accent-indigo bg-indigo-50 text-accent-indigo' :
                op === 'Skip' ? 'border-warning bg-amber-50 text-warning' :
                'border-border bg-bg-card text-text-secondary'
              }`}>
                Layer {i + 1}: {op}
              </div>
            </div>
          ))}
          <svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#D1D5DB" strokeWidth="2" /></svg>
          <div className="shrink-0 px-4 py-3 bg-bg-card rounded-xl border border-border text-sm font-mono text-text-secondary">
            GRU → Output (64)
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alpha Convergence */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Alpha Convergence</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <LineChart data={alphaData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              {OPS.map((op, i) => (
                <Line key={op} type="monotone" dataKey={op} stroke={OP_COLORS[i]}
                  strokeWidth={op === 'Linear' || op === 'Attention' ? 2.5 : 1.5}
                  dot={false} animationDuration={1200} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Performance Comparison */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">NAS vs Hand-Designed</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <BarChart data={compareData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="sharpe" name="Sharpe" radius={[6, 6, 0, 0]}>
                {compareData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#C15F3C' : '#D1D5DB'} />
                ))}
              </Bar>
              <Bar dataKey="return" name="Return %" radius={[6, 6, 0, 0]}>
                {compareData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#6366F1' : '#E5E7EB'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
