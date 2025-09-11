import * as React from "react";
import { cn } from "./cn";

/**
 * Badge component
 * @example
 * <Badge>New</Badge>
 */
export type BadgeProps = React.HTMLAttributes<HTMLSpanElement>;

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, ...props }, ref) => (
    <span
      ref={ref}
      className={cn("inline-flex items-center rounded-xl bg-accent px-2 py-1 text-xs text-bg", className)}
      {...props}
    />
  )
);
Badge.displayName = "Badge";
