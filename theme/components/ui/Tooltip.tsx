import * as React from "react";
import { cn } from "./cn";

/**
 * Tooltip component
 * @example
 * <Tooltip content="Info"><button>?</button></Tooltip>
 */
export interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactElement;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children }) => {
  const [open, setOpen] = React.useState(false);
  const id = React.useId();
  return (
    <span className="relative inline-block">
      {React.cloneElement(children, {
        'aria-describedby': id,
        onMouseEnter: () => setOpen(true),
        onMouseLeave: () => setOpen(false),
        onFocus: () => setOpen(true),
        onBlur: () => setOpen(false),
      })}
      {open && (
        <span
          role="tooltip"
          id={id}
          className={cn(
            "absolute z-10 mt-2 whitespace-nowrap rounded-xl bg-fg px-2 py-1 text-xs text-bg",
            "translate-x-[-50%] left-1/2"
          )}
        >
          {content}
        </span>
      )}
    </span>
  );
};
