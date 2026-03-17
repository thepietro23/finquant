import { motion } from 'framer-motion';
import { fadeSlideUp } from '../../lib/animations';
import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  cream?: boolean;
  noPad?: boolean;
}

export default function Card({ children, className, cream, noPad }: CardProps) {
  return (
    <motion.div
      variants={fadeSlideUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-40px' }}
      className={clsx(
        'rounded-2xl border border-border transition-all duration-300',
        'hover:shadow-[0_10px_25px_rgba(193,95,60,0.08),0_4px_10px_rgba(0,0,0,0.04)]',
        'hover:border-l-[3px] hover:border-l-primary hover:-translate-y-0.5',
        cream ? 'bg-bg-cream' : 'bg-white',
        noPad ? '' : 'p-6',
        className,
      )}
    >
      {children}
    </motion.div>
  );
}
