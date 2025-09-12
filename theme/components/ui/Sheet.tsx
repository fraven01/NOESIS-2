import * as React from "react";
import { cn } from "./cn";

/**
 * Sliding sheet component
 * @example
 * <Sheet open={open} onClose={...}>Content</Sheet>
 */
export interface SheetProps extends React.HTMLAttributes<HTMLDivElement> {
  open: boolean;
  onClose: () => void;
  side?: "left" | "right" | "top" | "bottom";
  ariaLabel?: string;
}

export const Sheet: React.FC<SheetProps> = ({
  open,
  onClose,
  side = "right",
  ariaLabel = "Sheet",
  className,
  children,
  ...props
}) => {
  const ref = React.useRef<HTMLDivElement>(null);
  const previouslyFocused = React.useRef<HTMLElement | null>(null);

  React.useEffect(() => {
    if (open) {
      previouslyFocused.current = document.activeElement as HTMLElement;
      ref.current?.focus();
    } else {
      previouslyFocused.current?.focus();
    }
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50"
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
    >
      <div className="absolute inset-0 bg-fg/50" onClick={onClose} aria-hidden="true" />
      <div
        ref={ref}
        tabIndex={-1}
        className={cn(
          "absolute flex flex-col bg-bg p-6 shadow-lg focus:outline-none",
          side === "right" && "top-0 right-0 h-full w-80",
          side === "left" && "top-0 left-0 h-full w-80",
          side === "top" && "top-0 left-0 w-full h-80",
          side === "bottom" && "bottom-0 left-0 w-full h-80",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </div>
  );
};
