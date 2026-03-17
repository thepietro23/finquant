import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

interface SectorData {
  name: string;
  value: number;
}

const COLORS = [
  '#C15F3C', '#6366F1', '#0D9488', '#16A34A', '#F59E0B',
  '#3B82F6', '#A34E30', '#8B5CF6', '#EC4899', '#14B8A6',
];

export default function SectorDonut({ data }: { data: SectorData[] }) {
  return (
    <ResponsiveContainer width="100%" height={280} minHeight={1}>
      <PieChart>
        <Pie
          data={data} cx="50%" cy="50%" innerRadius={65} outerRadius={100}
          dataKey="value" nameKey="name" paddingAngle={2}
          animationDuration={1000} animationBegin={200}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} strokeWidth={0} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12,
            boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          }}
          formatter={(value) => `${Number(value).toFixed(1)}%`}
        />
        <Legend
          layout="vertical" align="right" verticalAlign="middle"
          iconType="circle" iconSize={8}
          wrapperStyle={{ fontSize: 12, color: '#6B7280' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
