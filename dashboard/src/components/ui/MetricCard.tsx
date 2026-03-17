import { motion } from 'framer-motion';
import CountUp from 'react-countup';
import { fadeSlideUp } from '../../lib/animations';
import { valueBg, formatPct } from '../../lib/formatters';
import SparkLine from '../charts/SparkLine';

interface MetricCardProps {
  title: string;
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  change?: number;
  sparkData?: number[];
  icon?: React.ReactNode;
}

export default function MetricCard({
  title, value, decimals = 2, prefix = '', suffix = '',
  change, sparkData, icon,
}: MetricCardProps) {
  return (
    <motion.div
      variants={fadeSlideUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="bg-white rounded-2xl border border-border p-5 transition-all duration-300
        hover:shadow-[0_10px_25px_rgba(193,95,60,0.08)] hover:border-l-[3px] hover:border-l-primary hover:-translate-y-0.5"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-text-secondary">{title}</span>
        {icon && <span className="text-text-muted">{icon}</span>}
      </div>

      <div className="flex items-end gap-3 mb-3">
        <span className="text-3xl font-bold font-mono text-text">
          {prefix}
          <CountUp end={value} decimals={decimals} duration={1.2} separator="," preserveValue />
          {suffix}
        </span>
        {change !== undefined && (
          <span className={`text-sm font-medium px-2 py-0.5 rounded-full ${valueBg(change)}`}>
            {formatPct(change)}
          </span>
        )}
      </div>

      {sparkData && sparkData.length > 0 && (
        <div className="h-10">
          <SparkLine data={sparkData} />
        </div>
      )}
    </motion.div>
  );
}
