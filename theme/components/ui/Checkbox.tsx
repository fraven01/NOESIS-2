import * as React from "react";
import { cn } from "./cn";

/**
 * Checkbox component
 * @example
 * <Checkbox checked={checked} onChange={setChecked} />
 */
export type CheckboxProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, ...props }, ref) => (
    <input
      type="checkbox"
      ref={ref}
      className={cn(
        "h-4 w-4 rounded border border-muted text-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
        className
      )}
      {...props}
    />
  )
);
Checkbox.displayName = "Checkbox";
