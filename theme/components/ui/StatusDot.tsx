import * as React from "react";
import { cn } from "./cn";

/**
 * Colored status indicator dot
 * @example
 * <StatusDot status="success" />
 */
export interface StatusDotProps {
  status: "success" | "warning" | "error" | "neutral";
  className?: string;
}

const statusClasses: Record<StatusDotProps["status"], string> = {
  success: "bg-accent",
  warning: "bg-muted",
  error: "bg-danger",
  neutral: "bg-fg",
};

export const StatusDot: React.FC<StatusDotProps> = ({ status, className }) => (
  <span
    className={cn(
      "inline-block h-2 w-2 rounded-full",
      statusClasses[status],
      className
    )}
    aria-hidden="true"
  />
);
