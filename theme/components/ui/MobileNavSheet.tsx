import * as React from "react";
import { Sheet } from "./Sheet";
import { cn } from "./cn";

/**
 * Off-canvas navigation for mobile screens.
 * @example
 * <MobileNavSheet open={open} onClose={...} links={[{label:'Home',href:'/'}]} />
 */
export interface MobileNavSheetProps {
  open: boolean;
  onClose: () => void;
  links: { label: string; href: string }[];
  className?: string;
}

export const MobileNavSheet: React.FC<MobileNavSheetProps> = ({
  open,
  onClose,
  links,
  className,
}) => {
  const contentRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (open) {
      const previous = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = previous;
      };
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

  return (
    <Sheet
      open={open}
      onClose={onClose}
      side="left"
      className={cn("w-64", className)}
      ariaLabel="Navigation"
    >
      <div ref={contentRef} className="mt-4">
        <nav className="flex flex-col gap-2">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="rounded-lg px-4 py-2 text-lg text-fg hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              {l.label}
            </a>
          ))}
        </nav>
      </div>
    </Sheet>
  );
};
