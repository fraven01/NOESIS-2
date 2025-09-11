import * as React from "react";
import { cn } from "./cn";

/**
 * Toast notification
 * @example
 * <Toast open={open} onClose={() => setOpen(false)}>Saved</Toast>
 */
export interface ToastProps extends React.HTMLAttributes<HTMLDivElement> {
  open: boolean;
  onClose: () => void;
  autoHide?: number;
}

export const Toast: React.FC<ToastProps> = ({
  open,
  onClose,
  autoHide = 3000,
  className,
  children,
  ...props
}) => {
  React.useEffect(() => {
    if (!open) return;
    const t = setTimeout(onClose, autoHide);
    return () => clearTimeout(t);
  }, [open, autoHide, onClose]);
  if (!open) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "fixed bottom-4 right-4 rounded-xl bg-fg px-4 py-2 text-bg shadow", 
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
