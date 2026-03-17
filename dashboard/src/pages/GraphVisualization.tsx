/**
 * Graph Visualization — Interactive stock connectivity network
 *
 * Shows NIFTY 50 stocks as nodes connected by:
 *   - Sector edges (same sector)
 *   - Supply chain edges (business relationships)
 *   - Correlation edges (|corr| > 0.6)
 *
 * Node size = RL portfolio weight
 * Node color = sector
 * Edge color = relationship type
 */
import { useEffect, useState, useMemo, useRef } from 'react';
import { GitGraph } from 'lucide-react';
import { api } from '../lib/api';
import type { StockInfo } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import Badge from '../components/ui/Badge';

// ── Sector colors ──
const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Unknown': '#9CA3AF',
};

const EDGE_COLORS = {
  sector: '#C15F3C',
  supply: '#6366F1',
  correlation: '#0D9488',
};

// ── Supply chain relationships (from stocks.py) ──
const SUPPLY_CHAIN: [string, string][] = [
  ['TATASTEEL.NS', 'TATAMOTORS.NS'], ['TATASTEEL.NS', 'MARUTI.NS'],
  ['JSWSTEEL.NS', 'MARUTI.NS'], ['RELIANCE.NS', 'ONGC.NS'],
  ['RELIANCE.NS', 'BPCL.NS'], ['TCS.NS', 'INFY.NS'],
  ['HDFCBANK.NS', 'BAJFINANCE.NS'], ['ICICIBANK.NS', 'SBIN.NS'],
  ['HINDUNILVR.NS', 'ITC.NS'], ['BHARTIARTL.NS', 'TECHM.NS'],
  ['LT.NS', 'NTPC.NS'], ['LT.NS', 'POWERGRID.NS'],
];

interface GraphNode {
  id: string;
  ticker: string;
  sector: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  weight: number; // RL portfolio weight
  dailyReturn: number;
}

interface GraphEdge {
  source: string;
  target: string;
  type: 'sector' | 'supply' | 'correlation';
}

// ── Simple force simulation ──
function applyForces(nodes: GraphNode[], edges: GraphEdge[], width: number, height: number) {
  const alpha = 0.3;
  const repulsion = 800;
  const attraction = 0.005;
  const centerForce = 0.01;

  // Center gravity
  nodes.forEach(n => {
    n.vx += (width / 2 - n.x) * centerForce;
    n.vy += (height / 2 - n.y) * centerForce;
  });

  // Repulsion between all nodes
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = repulsion / (dist * dist);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      nodes[i].vx -= fx;
      nodes[i].vy -= fy;
      nodes[j].vx += fx;
      nodes[j].vy += fy;
    }
  }

  // Attraction along edges
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) return;
    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const force = dist * attraction;
    s.vx += dx * force;
    s.vy += dy * force;
    t.vx -= dx * force;
    t.vy -= dy * force;
  });

  // Apply velocity
  nodes.forEach(n => {
    n.vx *= 0.8; // damping
    n.vy *= 0.8;
    n.x += n.vx * alpha;
    n.y += n.vy * alpha;
    // Keep in bounds
    n.x = Math.max(40, Math.min(width - 40, n.x));
    n.y = Math.max(40, Math.min(height - 40, n.y));
  });
}

export default function GraphVisualization() {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showEdgeType, setShowEdgeType] = useState({ sector: true, supply: true, correlation: true });
  const svgRef = useRef<SVGSVGElement>(null);
  const animRef = useRef<number>(0);
  const [tick, setTick] = useState(0);

  const W = 900, H = 600;

  // Load stocks
  useEffect(() => {
    api.stocks().then(d => setStocks(d.stocks)).catch(() => {});
  }, []);

  // Build graph when stocks load
  useEffect(() => {
    if (stocks.length === 0) return;

    // Create nodes with random initial positions + mock RL weights
    const sectorGroups: Record<string, number[]> = {};
    const newNodes: GraphNode[] = stocks.map((s, i) => {
      if (!sectorGroups[s.sector]) sectorGroups[s.sector] = [];
      sectorGroups[s.sector].push(i);
      const angle = (i / stocks.length) * Math.PI * 2;
      const r = 150 + Math.random() * 100;
      return {
        id: s.ticker,
        ticker: s.ticker.replace('.NS', ''),
        sector: s.sector,
        x: W / 2 + Math.cos(angle) * r,
        y: H / 2 + Math.sin(angle) * r,
        vx: 0, vy: 0,
        weight: Math.round((Math.random() * 8 + 0.5) * 10) / 10, // RL weight 0.5-8.5%
        dailyReturn: Math.round((Math.random() - 0.45) * 6 * 100) / 100,
      };
    });

    // Build edges
    const newEdges: GraphEdge[] = [];

    // Sector edges — connect stocks in same sector
    Object.values(sectorGroups).forEach(indices => {
      for (let i = 0; i < indices.length; i++) {
        for (let j = i + 1; j < indices.length; j++) {
          newEdges.push({
            source: stocks[indices[i]].ticker,
            target: stocks[indices[j]].ticker,
            type: 'sector',
          });
        }
      }
    });

    // Supply chain edges
    SUPPLY_CHAIN.forEach(([s, t]) => {
      if (stocks.find(st => st.ticker === s) && stocks.find(st => st.ticker === t)) {
        newEdges.push({ source: s, target: t, type: 'supply' });
      }
    });

    // Correlation edges (mock — random high correlation pairs)
    const tickerList = stocks.map(s => s.ticker);
    for (let i = 0; i < 40; i++) {
      const a = tickerList[Math.floor(Math.random() * tickerList.length)];
      const b = tickerList[Math.floor(Math.random() * tickerList.length)];
      if (a !== b) {
        newEdges.push({ source: a, target: b, type: 'correlation' });
      }
    }

    setNodes(newNodes);
    setEdges(newEdges);
  }, [stocks]);

  // Force simulation animation
  useEffect(() => {
    if (nodes.length === 0) return;
    let frame = 0;
    const maxFrames = 150;

    function step() {
      applyForces(nodes, edges, W, H);
      setTick(t => t + 1);
      frame++;
      if (frame < maxFrames) {
        animRef.current = requestAnimationFrame(step);
      }
    }
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [nodes.length]); // Only re-run when nodes first populate

  // Filter edges by type
  const visibleEdges = edges.filter(e => showEdgeType[e.type]);

  const nodeMap = useMemo(() => new Map(nodes.map(n => [n.id, n])), [nodes, tick]);

  // Highlighted edges (connected to hovered/selected node)
  const highlightId = hovered || selected;
  const connectedNodes = useMemo(() => {
    if (!highlightId) return new Set<string>();
    const connected = new Set<string>();
    visibleEdges.forEach(e => {
      if (e.source === highlightId) connected.add(e.target);
      if (e.target === highlightId) connected.add(e.source);
    });
    return connected;
  }, [highlightId, visibleEdges, tick]);

  const selectedNode = selected ? nodeMap.get(selected) : null;

  // Stats
  const edgeCounts = {
    sector: edges.filter(e => e.type === 'sector').length,
    supply: edges.filter(e => e.type === 'supply').length,
    correlation: edges.filter(e => e.type === 'correlation').length,
  };

  return (
    <div>
      <PageHeader
        title="Graph Visualization"
        subtitle="Interactive NIFTY 50 stock network — edges by sector, supply chain, correlation. Node size = RL portfolio weight."
        icon={<GitGraph size={24} />}
      />

      {/* Edge type filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <span className="text-sm font-medium text-text-secondary">Show edges:</span>
        {(['sector', 'supply', 'correlation'] as const).map(type => (
          <button key={type}
            onClick={() => setShowEdgeType(p => ({ ...p, [type]: !p[type] }))}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              showEdgeType[type]
                ? 'border-current opacity-100'
                : 'border-border opacity-40'
            }`}
            style={{ color: EDGE_COLORS[type] }}
          >
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: EDGE_COLORS[type] }} />
            {type.charAt(0).toUpperCase() + type.slice(1)} ({edgeCounts[type]})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6">
        {/* Main Graph */}
        <Card noPad className="overflow-hidden">
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`}
            className="w-full bg-bg-card cursor-grab active:cursor-grabbing"
            style={{ minHeight: 500 }}>
            {/* Edges */}
            {visibleEdges.map((e, i) => {
              const s = nodeMap.get(e.source);
              const t = nodeMap.get(e.target);
              if (!s || !t) return null;
              const isHighlighted = highlightId && (e.source === highlightId || e.target === highlightId);
              return (
                <line key={i}
                  x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                  stroke={EDGE_COLORS[e.type]}
                  strokeWidth={isHighlighted ? 2 : 0.6}
                  opacity={highlightId ? (isHighlighted ? 0.8 : 0.08) : 0.25}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map(n => {
              const r = 6 + n.weight * 2; // Node size by RL weight
              const color = SECTOR_COLORS[n.sector] || '#9CA3AF';
              const isHighlighted = !highlightId || n.id === highlightId || connectedNodes.has(n.id);
              const isSelected = n.id === selected;
              return (
                <g key={n.id}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => setSelected(s => s === n.id ? null : n.id)}
                  className="cursor-pointer"
                >
                  {/* Glow ring for selected */}
                  {isSelected && (
                    <circle cx={n.x} cy={n.y} r={r + 6} fill="none"
                      stroke={color} strokeWidth={2} opacity={0.4} />
                  )}
                  {/* Node circle */}
                  <circle cx={n.x} cy={n.y} r={r}
                    fill={color} opacity={isHighlighted ? 0.9 : 0.2}
                    stroke={isSelected ? '#111827' : 'none'} strokeWidth={2}
                  />
                  {/* Label */}
                  {(r > 10 || hovered === n.id) && (
                    <text x={n.x} y={n.y - r - 4} textAnchor="middle"
                      fontSize={9} fontWeight={600} fill={isHighlighted ? '#374151' : '#D1D5DB'}
                      fontFamily="Inter">
                      {n.ticker}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </Card>

        {/* Right Panel — Details */}
        <div className="space-y-4">
          {/* Selected Node Info */}
          {selectedNode ? (
            <Card>
              <h3 className="font-display font-bold text-lg text-secondary mb-2">{selectedNode.ticker}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-secondary">Sector</span>
                  <Badge variant="neutral">{selectedNode.sector}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">RL Weight</span>
                  <span className="font-mono font-semibold">{selectedNode.weight}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Daily Return</span>
                  <span className={`font-mono font-semibold ${
                    selectedNode.dailyReturn >= 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {selectedNode.dailyReturn > 0 ? '+' : ''}{selectedNode.dailyReturn}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Connections</span>
                  <span className="font-mono">{connectedNodes.size}</span>
                </div>
              </div>

              {/* Connected stocks */}
              <h4 className="text-xs font-medium text-text-secondary mt-4 mb-2">Connected Stocks</h4>
              <div className="flex flex-wrap gap-1.5">
                {[...connectedNodes].slice(0, 12).map(id => {
                  const node = nodeMap.get(id);
                  if (!node) return null;
                  return (
                    <span key={id} className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-bg-card border border-border-light"
                      style={{ borderLeftColor: SECTOR_COLORS[node.sector] || '#9CA3AF', borderLeftWidth: 2 }}>
                      {node.ticker}
                    </span>
                  );
                })}
              </div>
            </Card>
          ) : (
            <Card cream>
              <p className="text-sm text-text-secondary text-center py-4">
                Click a node to view stock details and connections
              </p>
            </Card>
          )}

          {/* Legend */}
          <Card>
            <h3 className="font-semibold text-sm text-secondary mb-3">Sector Legend</h3>
            <div className="space-y-1.5">
              {Object.entries(SECTOR_COLORS).filter(([k]) => k !== 'Unknown').map(([sector, color]) => {
                const count = stocks.filter(s => s.sector === sector).length;
                if (count === 0) return null;
                return (
                  <div key={sector} className="flex items-center gap-2 text-xs">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="text-text-secondary flex-1">{sector}</span>
                    <span className="font-mono text-text-muted">{count}</span>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Graph Stats */}
          <Card>
            <h3 className="font-semibold text-sm text-secondary mb-3">Graph Stats</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-text-secondary">Total Nodes</span>
                <span className="font-mono font-semibold">{nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Total Edges</span>
                <span className="font-mono font-semibold">{edges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Visible Edges</span>
                <span className="font-mono font-semibold">{visibleEdges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Avg Connections</span>
                <span className="font-mono font-semibold">
                  {nodes.length > 0 ? (visibleEdges.length * 2 / nodes.length).toFixed(1) : 0}
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
