import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Target, BarChart3 } from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
} from 'recharts';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import { staggerContainer } from '../lib/animations';

// Mock training reward curve
function genRewardCurve() {
  const data = [];
  let reward = -50;
  for (let ep = 0; ep <= 200; ep += 2) {
    reward += (Math.random() - 0.35) * 8;
    reward = Math.min(reward, 120);
    data.push({
      episode: ep,
      ppo: Math.round(reward + (Math.random() - 0.5) * 15),
      sac: Math.round(reward * 0.9 + 10 + (Math.random() - 0.5) * 12),
    });
  }
  return data;
}

// Mock portfolio weights (treemap data)
const STOCKS = [
  'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
  'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'KOTAKBANK',
  'ITC', 'LT', 'AXISBANK', 'HCLTECH', 'MARUTI',
];

function genWeights() {
  const raw = STOCKS.map(() => Math.random() * 10 + 1);
  const sum = raw.reduce((a, b) => a + b, 0);
  return STOCKS.map((name, i) => ({
    name,
    weight: Math.round((raw[i] / sum) * 1000) / 10,
    dailyReturn: Math.round((Math.random() - 0.45) * 6 * 100) / 100,
  }));
}

export default function RlAgent() {
  const [agent, setAgent] = useState<'PPO' | 'SAC'>('PPO');
  const rewardData = useMemo(genRewardCurve, []);
  const weights = useMemo(genWeights, []);

  return (
    <div>
      <PageHeader
        title="RL Agent Monitor"
        subtitle="PPO + SAC deep reinforcement learning agents — portfolio management"
        icon={<Brain size={24} />}
      />

      {/* Agent Toggle + Metrics */}
      <div className="flex items-center gap-3 mb-6">
        {(['PPO', 'SAC'] as const).map(a => (
          <button key={a} onClick={() => setAgent(a)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              agent === a
                ? 'bg-primary text-white shadow-sm'
                : 'bg-bg-card text-text-secondary hover:bg-primary-subtle'
            }`}>
            {a}
          </button>
        ))}
      </div>

      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard title="Episode" value={200} decimals={0} icon={<Target size={18} />} />
        <MetricCard title="Avg Reward" value={78.4} decimals={1} change={0.15} icon={<Zap size={18} />} />
        <MetricCard title="Sharpe (Eval)" value={1.52} decimals={2} change={0.08} icon={<BarChart3 size={18} />} />
        <MetricCard title="Max Drawdown" value={-11.8} decimals={1} suffix="%" />
      </motion.div>

      {/* Training Reward Curve */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Training Progress — PPO vs SAC
        </h2>
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={rewardData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <defs>
              <linearGradient id="grad-ppo" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#C15F3C" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#C15F3C" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="grad-sac" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366F1" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#6366F1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="episode" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{
              background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12,
              borderLeft: '3px solid #C15F3C', boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
            }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 13 }} />
            <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C"
              strokeWidth={2.5} fill="url(#grad-ppo)" dot={false} animationDuration={1500} />
            <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1"
              strokeWidth={2} fill="url(#grad-sac)" dot={false} animationDuration={1500} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      {/* Portfolio Weights Bar Chart */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Portfolio Weights — {agent} Agent
        </h2>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={weights} margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} angle={-35} textAnchor="end" />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              tickFormatter={(v: number) => `${v}%`} />
            <Tooltip contentStyle={{
              background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12,
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
            }} formatter={(v) => `${v}%`} />
            <Bar dataKey="weight" name="Weight" radius={[6, 6, 0, 0]} animationDuration={800}>
              {weights.map((w, i) => (
                <Cell key={i} fill={w.dailyReturn >= 0 ? '#C15F3C' : '#6366F1'} opacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex items-center gap-4 mt-3 text-xs text-text-secondary">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-primary" /> Positive daily return
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-accent-indigo" /> Negative daily return
          </span>
        </div>
      </Card>
    </div>
  );
}
