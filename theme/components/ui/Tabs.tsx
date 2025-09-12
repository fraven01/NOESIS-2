import * as React from "react";
import { cn } from "./cn";

/**
 * Tabs component
 * @example
 * <Tabs defaultValue="one"><TabsList>...</TabsList></Tabs>
 */
interface TabsContextProps {
  value: string;
  setValue: (v: string) => void;
}
const TabsContext = React.createContext<TabsContextProps | null>(null);

export interface TabsProps {
  defaultValue: string;
  children: React.ReactNode;
}

export const Tabs: React.FC<TabsProps> = ({ defaultValue, children }) => {
  const [value, setValue] = React.useState(defaultValue);
  return (
    <TabsContext.Provider value={{ value, setValue }}>
      {children}
    </TabsContext.Provider>
  );
};

export const TabsList: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  ...props
}) => {
  const ref = React.useRef<HTMLDivElement>(null);
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const tabs = Array.from(
      ref.current.querySelectorAll<HTMLElement>("[role=tab]")
    );
    const currentIndex = tabs.indexOf(document.activeElement as HTMLElement);
    if (e.key === "ArrowRight") {
      const next = tabs[(currentIndex + 1) % tabs.length];
      next.focus();
      next.click();
      e.preventDefault();
    }
    if (e.key === "ArrowLeft") {
      const prev = tabs[(currentIndex - 1 + tabs.length) % tabs.length];
      prev.focus();
      prev.click();
      e.preventDefault();
    }
  };
  return (
    <div
      ref={ref}
      role="tablist"
      onKeyDown={onKeyDown}
      className={cn("flex gap-2", className)}
      {...props}
    />
  );
};

export interface TabsTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
}

export const TabsTrigger = React.forwardRef<HTMLButtonElement, TabsTriggerProps>(
  ({ className, value, ...props }, ref) => {
    const ctx = React.useContext(TabsContext);
    if (!ctx) throw new Error("Tabs components must be used within <Tabs>");
    const selected = ctx.value === value;
    return (
      <button
        ref={ref}
        role="tab"
        aria-selected={selected}
        onClick={() => ctx.setValue(value)}
        className={cn(
          "rounded-xl px-4 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
          selected ? "bg-accent text-bg" : "bg-muted text-fg",
          className
        )}
        {...props}
      />
    );
  }
);
TabsTrigger.displayName = "TabsTrigger";

export interface TabsContentProps
  extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

export const TabsContent = React.forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value, ...props }, ref) => {
    const ctx = React.useContext(TabsContext);
    if (!ctx) throw new Error("Tabs components must be used within <Tabs>");
    return ctx.value === value ? (
      <div
        ref={ref}
        role="tabpanel"
        className={cn(className)}
        {...props}
      />
    ) : null;
  }
);
TabsContent.displayName = "TabsContent";
