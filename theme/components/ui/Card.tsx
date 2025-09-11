import * as React from "react";
import { cn } from "./cn";

/**
 * Card container
 * @example
 * <Card>...</Card>
 */
export type CardProps = React.HTMLAttributes<HTMLDivElement>;

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("rounded-2xl border border-muted bg-bg p-6 shadow", className)}
      {...props}
    />
  )
);
Card.displayName = "Card";
