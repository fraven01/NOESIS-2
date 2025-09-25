import * as React from "react";
import { cn } from "./cn";

/**
 * Form label
 * @example
 * <Label htmlFor="email">Email</Label>
 */
export type LabelProps = React.LabelHTMLAttributes<HTMLLabelElement>;

export const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn("mb-1 block text-sm font-medium text-fg", className)}
      {...props}
    />
  )
);
Label.displayName = "Label";
