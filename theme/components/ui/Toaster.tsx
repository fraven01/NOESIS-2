import * as React from "react";
import { Toast } from "./Toast";

/**
 * Toast manager that provides a context to dispatch notifications.
 * @example
 * <Toaster><App /></Toaster>
 */
export interface ToastItem {
  id: number;
  message: string;
}

interface ToasterContextValue {
  push: (message: string) => void;
}

const ToasterContext = React.createContext<ToasterContextValue | null>(null);

export const useToaster = () => {
  const ctx = React.useContext(ToasterContext);
  if (!ctx) throw new Error("useToaster must be used within Toaster");
  return ctx;
};

export const Toaster: React.FC<React.PropsWithChildren> = ({ children }) => {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);
  const previouslyFocused = React.useRef<HTMLElement | null>(null);

  const push = (message: string) => {
    previouslyFocused.current = document.activeElement as HTMLElement;
    const id = Date.now();
    setToasts((t) => [...t, { id, message }]);
  };

  const close = (id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
    previouslyFocused.current?.focus();
  };

  return (
    <ToasterContext.Provider value={{ push }}>
      {children}
      <div
        role="region"
        aria-live="polite"
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2"
      >
        {toasts.map((t) => (
          <Toast
            key={t.id}
            open
            onClose={() => close(t.id)}
            onKeyDown={(e) => {
              if (e.key === "Escape") close(t.id);
            }}
            tabIndex={0}
          >
            {t.message}
          </Toast>
        ))}
      </div>
    </ToasterContext.Provider>
  );
};
