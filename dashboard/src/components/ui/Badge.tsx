import { clsx } from 'clsx';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'profit' | 'loss' | 'warning' | 'info' | 'neutral';
}

const variants = {
  profit: 'bg-profit-light text-profit',
  loss: 'bg-loss-light text-loss',
  warning: 'bg-amber-50 text-warning',
  info: 'bg-blue-50 text-info',
  neutral: 'bg-bg-card text-text-secondary',
};

export default function Badge({ children, variant = 'neutral' }: BadgeProps) {
  return (
    <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', variants[variant])}>
      {children}
    </span>
  );
}
