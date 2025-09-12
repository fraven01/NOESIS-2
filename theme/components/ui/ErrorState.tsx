import * as React from "react";
import { cn } from "./cn";
import { Button } from "./Button";

/**
 * Error state display
 * @example
 * <ErrorState title="Error" text="Something went wrong" cta={{label:'Retry',onClick:()=>{}}} />
 */
export interface ErrorStateProps {
  icon?: React.ReactNode;
  title: string;
  text: string;
  cta?: { label: string; onClick: () => void };
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  icon,
  title,
  text,
  cta,
  className,
}) => (
  <div
    role="alert"
    className={cn(
      "flex flex-col items-center justify-center gap-4 rounded-2xl border border-danger p-8 text-center text-danger",
      className
    )}
  >
    {icon && <div aria-hidden="true" className="text-4xl">{icon}</div>}
    <h2 className="text-lg font-semibold">{title}</h2>
    <p className="text-sm">{text}</p>
    {cta && (
      <Button onClick={cta.onClick} variant="primary">
        {cta.label}
      </Button>
    )}
  </div>
);
