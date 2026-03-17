import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
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

function useAnimatedNumber(end: number, decimals: number, duration = 1200) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const currentRef = useRef(0);

  useEffect(() => {
    const startVal = currentRef.current;
    startRef.current = performance.now();
    function tick(now: number) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = startVal + (end - startVal) * eased;
      currentRef.current = val;
      setDisplay(val);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [end, duration]);

  return display.toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export default function MetricCard({
  title, value, decimals = 2, prefix = '', suffix = '',
  change, sparkData, icon,
}: MetricCardProps) {
  const animated = useAnimatedNumber(value, decimals);

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
          {prefix}{animated}{suffix}
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
