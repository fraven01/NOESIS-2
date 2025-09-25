import * as React from "react";
import { cn } from "./cn";

/**
 * Select component
 * @example
 * <Select><option>One</option></Select>
 */
export type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        "w-full rounded-xl border border-muted bg-bg px-4 py-2 text-fg shadow-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent",
        className
      )}
      {...props}
    >
      {children}
    </select>
  )
);
Select.displayName = "Select";
