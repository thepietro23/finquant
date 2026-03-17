import { useMemo } from 'react';
import { Users, Shield, Lock, Activity } from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar,
} from 'recharts';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import { staggerContainer } from '../lib/animations';
import { motion } from 'framer-motion';

const CLIENTS = ['Banking+Finance', 'IT+Telecom', 'Pharma+FMCG', 'Energy+Auto+Others'];
const CLIENT_COLORS = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B'];

function genConvergence() {
  const data = [];
  const sharpes = [0.3, 0.25, 0.35, 0.2];
  for (let round = 0; round <= 50; round++) {
    const entry: Record<string, number> = { round };
    sharpes.forEach((_s, i) => {
      sharpes[i] += (Math.random() - 0.3) * 0.06;
      entry[CLIENTS[i]] = Math.round(sharpes[i] * 100) / 100;
    });
    entry['FedProx'] = Math.round((sharpes.reduce((a, b) => a + b) / 4 + 0.1) * 100) / 100;
    entry['FedAvg'] = Math.round((sharpes.reduce((a, b) => a + b) / 4 + 0.05) * 100) / 100;
    data.push(entry);
  }
  return data;
}

export default function Federated() {
  const convData = useMemo(genConvergence, []);

  const fairnessData = CLIENTS.map((c) => ({
    client: c.split('+')[0],
    withFL: 1.2 + Math.random() * 0.6,
    withoutFL: 0.7 + Math.random() * 0.5,
  }));

  return (
    <div>
      <PageHeader
        title="Federated Learning"
        subtitle="FedAvg/FedProx + DP-SGD — 4 sector-wise clients, privacy-preserving"
        icon={<Users size={24} />}
      />

      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard title="FL Rounds" value={50} decimals={0} icon={<Activity size={18} />} />
        <MetricCard title="Privacy ε" value={5.2} decimals={1} suffix="/8.0" icon={<Lock size={18} />} />
        <MetricCard title="Global Sharpe" value={1.48} decimals={2} change={0.12} icon={<Shield size={18} />} />
        <MetricCard title="Clients" value={4} decimals={0} />
      </motion.div>

      {/* Client Info Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {CLIENTS.map((c, i) => (
          <Card key={c}>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: CLIENT_COLORS[i] }} />
              <span className="text-sm font-medium text-text">{c.split('+').join(' + ')}</span>
            </div>
            <p className="text-xs text-text-secondary">Client {i}</p>
            <p className="text-lg font-mono font-bold text-text mt-1">
              {[10, 6, 8, 23][i]} stocks
            </p>
          </Card>
        ))}
      </div>

      {/* Convergence Curves */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Convergence — Sharpe vs FL Rounds
        </h2>
        <ResponsiveContainer width="100%" height={350} minHeight={1}>
          <LineChart data={convData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="round" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="FedProx" stroke="#C15F3C" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="FedAvg" stroke="#6366F1" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
            {CLIENTS.map((c, i) => (
              <Line key={c} type="monotone" dataKey={c} stroke={CLIENT_COLORS[i]}
                strokeWidth={1} strokeOpacity={0.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Fairness Comparison */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Client Fairness — With FL vs Without FL
        </h2>
        <ResponsiveContainer width="100%" height={280} minHeight={1}>
          <BarChart data={fairnessData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="client" tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              label={{ value: 'Sharpe Ratio', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#9CA3AF' } }} />
            <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="withFL" name="With FL" fill="#C15F3C" radius={[6, 6, 0, 0]} />
            <Bar dataKey="withoutFL" name="Without FL" fill="#D1D5DB" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
