import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LayoutDashboard, PieChart, Network, Brain, AlertTriangle,
  FlaskConical, Users, Atom, MessageSquare, GitGraph,
  Settings, ExternalLink, ChevronLeft, ChevronRight,
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/gnn', icon: Network, label: 'GNN Insights' },
  { to: '/rl', icon: Brain, label: 'RL Agent' },
  { to: '/stress', icon: AlertTriangle, label: 'Stress Testing' },
  { to: '/nas', icon: FlaskConical, label: 'NAS Lab' },
  { to: '/fl', icon: Users, label: 'Federated' },
  { to: '/quantum', icon: Atom, label: 'Quantum' },
  { to: '/sentiment', icon: MessageSquare, label: 'Sentiment' },
  { to: '/graph', icon: GitGraph, label: 'Graph Viz' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 260 }}
      transition={{ type: 'spring', stiffness: 200, damping: 25 }}
      className="h-screen sticky top-0 bg-white border-r border-border flex flex-col z-30"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-border">
        <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center shrink-0">
          <span className="text-white font-bold text-sm font-display">FQ</span>
        </div>
        {!collapsed && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="overflow-hidden">
            <h1 className="font-display font-bold text-base text-secondary leading-tight">FINQUANT</h1>
            <p className="text-[10px] text-text-muted font-medium tracking-wider">NEXUS v4</p>
          </motion.div>
        )}
      </div>

      {/* Nav Items */}
      <nav className="flex-1 py-3 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 mx-2 my-0.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-primary-subtle text-primary border-l-[3px] border-primary'
                  : 'text-text-secondary hover:bg-bg-card hover:text-text'
              }`
            }
          >
            <Icon size={20} className="shrink-0" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Bottom */}
      <div className="border-t border-border py-3">
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
          className="flex items-center gap-3 mx-2 px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:bg-bg-card transition-colors">
          <ExternalLink size={20} className="shrink-0" />
          {!collapsed && <span>API Docs</span>}
        </a>
        <div className="flex items-center gap-3 mx-2 px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:bg-bg-card cursor-pointer transition-colors">
          <Settings size={20} className="shrink-0" />
          {!collapsed && <span>Settings</span>}
        </div>
      </div>

      {/* Collapse Toggle */}
      <button
        onClick={() => setCollapsed(c => !c)}
        className="absolute -right-3 top-20 w-6 h-6 bg-white border border-border rounded-full flex items-center justify-center shadow-sm hover:bg-bg-card transition-colors"
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>
    </motion.aside>
  );
}
