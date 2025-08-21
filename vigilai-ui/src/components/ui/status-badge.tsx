import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: "idle" | "active" | "error" | "training" | "detecting" | "evaluating";
  className?: string;
}

const statusConfig = {
  idle: {
    label: "IDLE",
    className: "status-idle"
  },
  active: {
    label: "ACTIVE", 
    className: "status-active"
  },
  error: {
    label: "ERROR",
    className: "status-error"
  },
  training: {
    label: "TRAINING",
    className: "status-active"
  },
  detecting: {
    label: "DETECTING", 
    className: "status-active"
  },
  evaluating: {
    label: "EVALUATING",
    className: "status-active"
  }
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config = statusConfig[status];
  
  return (
    <span 
      className={cn(
        "px-3 py-1 text-xs font-mono font-bold tracking-wider rounded border",
        config.className,
        className
      )}
    >
      {config.label}
    </span>
  );
}