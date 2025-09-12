import * as React from "react";
import { cn } from "../ui/cn";
import { Label } from "../ui/Label";
import { HelperText } from "./HelperText";
import { ErrorText } from "./ErrorText";

/**
 * Wraps a form control with label and messages.
 * @example
 * <FormField name="email" label="Email" helperText="Required"><Input /></FormField>
 */
export interface FormFieldProps extends React.HTMLAttributes<HTMLDivElement> {
  name: string;
  label: string;
  helperText?: string;
  error?: string;
  children: React.ReactElement;
}

export const FormField: React.FC<FormFieldProps> = ({
  name,
  label,
  helperText,
  error,
  children,
  className,
  ...props
}) => {
  const helperId = helperText ? `${name}-helper` : undefined;
  const errorId = error ? `${name}-error` : undefined;
  const describedBy = [errorId, helperId].filter(Boolean).join(" ") || undefined;
  const field = React.cloneElement(children, {
    id: name,
    name,
    className: cn("h-11", children.props.className),
    "aria-describedby": describedBy,
    "aria-invalid": !!error || undefined,
  });

  return (
    <div className={cn("flex flex-col gap-1", className)} {...props}>
      <Label htmlFor={name}>{label}</Label>
      {field}
      {helperText && !error && <HelperText id={helperId}>{helperText}</HelperText>}
      {error && <ErrorText id={errorId}>{error}</ErrorText>}
    </div>
  );
};
