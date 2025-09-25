import * as React from "react";
import { cn } from "../ui/cn";

/**
 * Inline error message for a field.
 * @example
 * <ErrorText id="email-error">Email is required</ErrorText>
 */
export type ErrorTextProps = React.HTMLAttributes<HTMLParagraphElement>;

export const ErrorText = React.forwardRef<HTMLParagraphElement, ErrorTextProps>(
  ({ className, ...props }, ref) => (
    <p ref={ref} role="alert" className={cn("text-sm text-danger", className)} {...props} />
  )
);
ErrorText.displayName = "ErrorText";
