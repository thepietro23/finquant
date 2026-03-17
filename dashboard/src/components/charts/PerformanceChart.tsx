import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

interface DataPoint {
  date: string;
  portfolio: number;
  nifty: number;
}

interface PerformanceChartProps {
  data: DataPoint[];
  height?: number;
}

export default function PerformanceChart({ data, height = 350 }: PerformanceChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height} minHeight={1}>
      <AreaChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
        <defs>
          <linearGradient id="grad-portfolio" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#C15F3C" stopOpacity={0.2} />
            <stop offset="100%" stopColor="#C15F3C" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="date" tick={{ fontSize: 12, fill: '#9CA3AF' }}
          axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 12, fill: '#9CA3AF' }}
          axisLine={false} tickLine={false}
          tickFormatter={(v: number) => `${v.toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{
            background: '#fff', border: '1px solid #E5E7EB',
            borderRadius: 12, borderLeft: '3px solid #C15F3C',
            boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          }}
          labelStyle={{ color: '#6B7280', fontSize: 12 }}
        />
        <Legend
          wrapperStyle={{ fontSize: 13, color: '#6B7280' }}
          iconType="circle" iconSize={8}
        />
        <Area
          type="monotone" dataKey="portfolio" name="FINQUANT Portfolio"
          stroke="#C15F3C" strokeWidth={2.5} fill="url(#grad-portfolio)"
          dot={false} animationDuration={1500}
        />
        <Area
          type="monotone" dataKey="nifty" name="NIFTY 50"
          stroke="#D1D5DB" strokeWidth={1.5} strokeDasharray="5 5"
          fill="none" dot={false} animationDuration={1500}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
