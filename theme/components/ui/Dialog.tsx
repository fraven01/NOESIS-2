import * as React from "react";
import { cn } from "./cn";

/**
 * Accessible modal dialog with focus trap
 * @example
 * <Dialog open={open} onClose={() => setOpen(false)}>Content</Dialog>
 */
export interface DialogProps extends React.HTMLAttributes<HTMLDivElement> {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
  ariaLabel?: string;
}

export const Dialog: React.FC<DialogProps> = ({
  open,
  onClose,
  className,
  children,
  ariaLabel = "Dialog",
  ...props
}) => {
  const contentRef = React.useRef<HTMLDivElement>(null);
  const previouslyFocused = React.useRef<HTMLElement | null>(null);

  React.useEffect(() => {
    if (open) {
      previouslyFocused.current = document.activeElement as HTMLElement;
      contentRef.current?.focus();
    } else {
      previouslyFocused.current?.focus();
    }
  }, [open]);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
      if (e.key === "Tab" && contentRef.current) {
        const focusable = contentRef.current.querySelectorAll<HTMLElement>(
          'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
        );
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          last.focus();
          e.preventDefault();
        } else if (!e.shiftKey && document.activeElement === last) {
          first.focus();
          e.preventDefault();
        }
      }
    };
    if (open) {
      document.addEventListener("keydown", handler);
      return () => document.removeEventListener("keydown", handler);
    }
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      className="fixed inset-0 z-50 flex items-center justify-center"
    >
      <div
        className="absolute inset-0 bg-fg/50" onClick={onClose} aria-hidden="true" />
      <div
        ref={contentRef}
        tabIndex={-1}
        className={cn(
          "relative max-w-lg rounded-2xl bg-bg p-6 shadow-lg focus:outline-none",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </div>
  );
};
