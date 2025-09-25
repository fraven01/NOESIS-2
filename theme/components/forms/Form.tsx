import * as React from "react";
import { cn } from "../ui/cn";

/**
 * Semantic form wrapper.
 * @example
 * <Form onSubmit={console.log}>...</Form>
 */
export type FormProps = React.FormHTMLAttributes<HTMLFormElement>;

export const Form = React.forwardRef<HTMLFormElement, FormProps>(
  ({ className, ...props }, ref) => (
    <form
      ref={ref}
      noValidate
      className={cn("flex flex-col gap-4", className)}
      {...props}
    />
  )
);
Form.displayName = "Form";
