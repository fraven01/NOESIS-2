import * as React from "react";
import { cn } from "./cn";
import { Button } from "./Button";

/**
 * Empty state display
 * @example
 * <EmptyState icon={<span />} title="No items" text="Add one" cta={{label:'Add',onClick:()=>{}}} />
 */
export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  text: string;
  cta?: { label: string; onClick: () => void };
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  text,
  cta,
  className,
}) => (
  <div
    className={cn(
      "flex flex-col items-center justify-center gap-4 rounded-2xl border border-muted p-8 text-center text-fg",
      className
    )}
  >
    {icon && <div aria-hidden="true" className="text-4xl">{icon}</div>}
    <h2 className="text-lg font-semibold">{title}</h2>
    <p className="text-sm text-muted">{text}</p>
    {cta && (
      <Button onClick={cta.onClick} variant="primary">
        {cta.label}
      </Button>
    )}
  </div>
);
