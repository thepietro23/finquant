import { useState } from 'react';
import { Atom, Play, Cpu, Zap } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { QAOAResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import { staggerContainer } from '../lib/animations';
import { motion } from 'framer-motion';

export default function Quantum() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QAOAResponse | null>(null);
  const [nAssets, setNAssets] = useState(6);
  const [kSelect, setKSelect] = useState(3);
  const [layers, setLayers] = useState(2);

  async function runQAOA() {
    const k = Math.min(kSelect, nAssets);
    setLoading(true);
    setError(null);
    try {
      const res = await api.qaoa(nAssets, k, layers, 512, 0.5);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'QAOA failed — is the backend running?');
    } finally {
      setLoading(false);
    }
  }

  // Comparison data for chart
  const compareData = result ? [
    { name: 'Quantum (QAOA)', sharpe: result.quantum_sharpe },
    { name: 'Classical (Brute-Force)', sharpe: result.classical_sharpe },
  ] : [];

  // Weight comparison
  const weightData = result ? result.quantum_weights.map((w, i) => ({
    asset: `Asset ${result.quantum_assets[i] ?? i}`,
    quantum: Math.round(w * 100) / 100,
    classical: Math.round((result.classical_weights[i] ?? 0) * 100) / 100,
  })) : [];

  return (
    <div>
      <PageHeader
        title="Quantum Lab"
        subtitle="QAOA portfolio selection — Qiskit simulator + classical benchmark"
        icon={<Atom size={24} />}
      />

      {/* Controls */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">N Assets</label>
            <input type="number" value={nAssets} onChange={e => setNAssets(+e.target.value)}
              min={2} max={12}
              className="w-20 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">K Select</label>
            <input type="number" value={kSelect} onChange={e => setKSelect(+e.target.value)}
              min={1} max={nAssets}
              className="w-20 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">QAOA Layers</label>
            <input type="number" value={layers} onChange={e => setLayers(+e.target.value)}
              min={1} max={5}
              className="w-20 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <button onClick={runQAOA} disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50">
            <Play size={16} />
            {loading ? 'Optimizing...' : 'Run QAOA'}
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-loss">{error}</p>}
      </Card>

      {result && (
        <>
          <motion.div variants={staggerContainer} initial="hidden" animate="visible"
            className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <MetricCard title="Quantum Sharpe" value={result.quantum_sharpe} decimals={4}
              icon={<Atom size={18} />} />
            <MetricCard title="Classical Sharpe" value={result.classical_sharpe} decimals={4}
              icon={<Cpu size={18} />} />
            <MetricCard title="Qubits" value={result.n_qubits} decimals={0}
              icon={<Zap size={18} />} />
            <MetricCard title="Function Evals" value={result.n_function_evals} decimals={0} />
          </motion.div>

          {/* Circuit Diagram (simplified SVG) */}
          <Card className="mb-6">
            <h2 className="font-display font-bold text-lg text-secondary mb-4">
              QAOA Circuit — {result.n_qubits} qubits, {layers} layers
            </h2>
            <div className="overflow-x-auto bg-bg-card rounded-xl p-4">
              <svg viewBox={`0 0 ${300 + layers * 120} ${result.n_qubits * 40 + 20}`}
                className="w-full max-h-48">
                {/* Qubit wires */}
                {Array.from({ length: result.n_qubits }).map((_, q) => (
                  <g key={q}>
                    <text x={10} y={30 + q * 40} fontSize={11} fill="#6B7280" fontFamily="JetBrains Mono">
                      q{q}
                    </text>
                    <line x1={35} y1={27 + q * 40} x2={280 + layers * 120} y2={27 + q * 40}
                      stroke="#D1D5DB" strokeWidth={1.5} />
                  </g>
                ))}
                {/* H gates */}
                {Array.from({ length: result.n_qubits }).map((_, q) => (
                  <rect key={`h${q}`} x={50} y={14 + q * 40} width={26} height={26}
                    rx={4} fill="#C15F3C" opacity={0.9} />
                ))}
                {Array.from({ length: result.n_qubits }).map((_, q) => (
                  <text key={`ht${q}`} x={63} y={31 + q * 40} textAnchor="middle"
                    fontSize={12} fill="white" fontWeight={700}>H</text>
                ))}
                {/* QAOA layers */}
                {Array.from({ length: layers }).map((_, l) => (
                  <g key={`l${l}`}>
                    {/* Cost unitary */}
                    <rect x={100 + l * 120} y={8} width={40}
                      height={result.n_qubits * 40} rx={6} fill="#6366F1" opacity={0.15}
                      stroke="#6366F1" strokeWidth={1} />
                    <text x={120 + l * 120} y={result.n_qubits * 20 + 12} textAnchor="middle"
                      fontSize={9} fill="#6366F1" fontWeight={600}>Cost γ{l + 1}</text>
                    {/* Mixer */}
                    <rect x={155 + l * 120} y={8} width={40}
                      height={result.n_qubits * 40} rx={6} fill="#C15F3C" opacity={0.15}
                      stroke="#C15F3C" strokeWidth={1} />
                    <text x={175 + l * 120} y={result.n_qubits * 20 + 12} textAnchor="middle"
                      fontSize={9} fill="#C15F3C" fontWeight={600}>Mix β{l + 1}</text>
                  </g>
                ))}
                {/* Measurement */}
                {Array.from({ length: result.n_qubits }).map((_, q) => (
                  <g key={`m${q}`}>
                    <rect x={220 + layers * 120} y={14 + q * 40} width={26} height={26}
                      rx={4} fill="none" stroke="#111827" strokeWidth={1.5} />
                    <path d={`M${227 + layers * 120} ${32 + q * 40} A6 6 0 0 1 ${239 + layers * 120} ${32 + q * 40}`}
                      fill="none" stroke="#111827" strokeWidth={1.2} />
                    <line x1={233 + layers * 120} y1={32 + q * 40} x2={236 + layers * 120} y2={20 + q * 40}
                      stroke="#111827" strokeWidth={1.2} />
                  </g>
                ))}
              </svg>
            </div>
            <div className="flex items-center gap-4 mt-3 text-xs text-text-secondary">
              <span>Best bitstring: <code className="font-mono bg-bg-card px-2 py-0.5 rounded">{result.best_bitstring}</code></span>
              <span>Selected assets: <code className="font-mono bg-bg-card px-2 py-0.5 rounded">[{result.quantum_assets.join(', ')}]</code></span>
            </div>
          </Card>

          {/* Sharpe Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <h2 className="font-display font-bold text-lg text-secondary mb-4">Sharpe Comparison</h2>
              <ResponsiveContainer width="100%" height={250} minHeight={1}>
                <BarChart data={compareData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6B7280' }}
                    axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
                  <Bar dataKey="sharpe" name="Sharpe Ratio" radius={[8, 8, 0, 0]} animationDuration={800}>
                    <Cell fill="#C15F3C" />
                    <Cell fill="#6366F1" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <h2 className="font-display font-bold text-lg text-secondary mb-4">Portfolio Weights</h2>
              <ResponsiveContainer width="100%" height={250} minHeight={1}>
                <BarChart data={weightData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="asset" tick={{ fontSize: 11, fill: '#6B7280' }}
                    axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="quantum" name="Quantum" fill="#C15F3C" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="classical" name="Classical" fill="#6366F1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {!result && (
        <Card cream className="text-center py-16">
          <Atom size={48} className="mx-auto text-primary mb-4 opacity-40" />
          <p className="text-text-secondary">Configure parameters and click <strong>Run QAOA</strong> to start quantum optimization</p>
        </Card>
      )}
    </div>
  );
}
