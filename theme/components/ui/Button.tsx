import * as React from "react";
import { cn } from "./cn";

/**
 * Button component
 * @example
 * <Button variant="primary">Save</Button>
 */
export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "destructive";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
};

const variantClasses: Record<NonNullable<ButtonProps["variant"]>, string> = {
  primary: "bg-accent text-bg hover:bg-accent/90",
  secondary: "bg-muted text-fg hover:bg-muted/80",
  ghost: "bg-transparent text-fg hover:bg-muted/20",
  destructive: "bg-danger text-bg hover:bg-danger/90",
};

const sizeClasses: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm: "px-3 py-1 text-sm",
  md: "px-4 py-2 text-base",
  lg: "px-6 py-3 text-lg",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      loading = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-xl font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none transition-colors",
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        aria-busy={loading || undefined}
        disabled={disabled || loading}
        {...props}
      >
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";
