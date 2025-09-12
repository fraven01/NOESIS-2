import * as React from "react";
import { cn } from "../ui/cn";

/**
 * Contextual helper text for a field.
 * @example
 * <HelperText id="email-helper">We'll never share your email.</HelperText>
 */
export type HelperTextProps = React.HTMLAttributes<HTMLParagraphElement>;

export const HelperText = React.forwardRef<HTMLParagraphElement, HelperTextProps>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn("text-sm text-muted", className)} {...props} />
  )
);
HelperText.displayName = "HelperText";
