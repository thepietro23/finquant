import { motion } from 'framer-motion';

interface PageHeaderProps {
  title: string;
  subtitle: string;
  icon?: React.ReactNode;
}

export default function PageHeader({ title, subtitle, icon }: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-6"
    >
      <div className="flex items-center gap-3 mb-1">
        {icon && <span className="text-primary">{icon}</span>}
        <h1 className="font-display font-bold text-2xl text-secondary">{title}</h1>
      </div>
      <p className="text-sm text-text-secondary">{subtitle}</p>
    </motion.div>
  );
}
