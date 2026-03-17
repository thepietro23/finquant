import { ResponsiveContainer, AreaChart, Area } from 'recharts';

interface SparkLineProps {
  data: number[];
  color?: string;
}

export default function SparkLine({ data, color = '#C15F3C' }: SparkLineProps) {
  const chartData = data.map((v, i) => ({ i, v }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <defs>
          <linearGradient id={`spark-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone" dataKey="v" stroke={color} strokeWidth={1.5}
          fill={`url(#spark-${color})`} dot={false} animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
