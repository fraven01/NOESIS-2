import * as React from "react";
import { cn } from "./cn";

/**
 * Textarea component
 * @example
 * <Textarea rows={3} />
 */
export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded-xl border border-muted bg-bg px-4 py-2 text-fg shadow-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent",
        className
      )}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";
