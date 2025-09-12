import * as React from "react";
import { cn } from "./cn";

/**
 * Text input component
 * @example
 * <Input placeholder="Email" />
 */
export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "w-full rounded-xl border border-muted bg-bg px-4 py-2 text-fg shadow-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";
