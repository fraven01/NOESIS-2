import * as React from "react";
import { cn } from "./cn";

/**
 * Switch component
 * @example
 * <Switch checked={on} onCheckedChange={setOn} />
 */
export interface SwitchProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onChange'> {
  checked: boolean;
  onCheckedChange?: (checked: boolean) => void;
}

export const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
  ({ className, checked, onCheckedChange, disabled, ...props }, ref) => {
    const toggle = () => {
      if (disabled) return;
      onCheckedChange?.(!checked);
    };
    return (
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        ref={ref}
        onClick={toggle}
        className={cn(
          "inline-flex h-6 w-10 items-center rounded-xl border border-muted bg-bg p-1 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
          checked ? "bg-accent" : "bg-muted",
          disabled && "opacity-50",
          className
        )}
        disabled={disabled}
        {...props}
      >
        <span
          className={cn(
            "block h-4 w-4 rounded-xl bg-bg transition-transform",
            checked && "translate-x-4"
          )}
        />
      </button>
    );
  }
);
Switch.displayName = "Switch";
