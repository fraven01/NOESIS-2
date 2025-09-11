import * as React from "react";
import { cn } from "./cn";
import { Button } from "./Button";

/**
 * Pagination controls
 * @example
 * <Pagination current={1} total={5} onChange={setPage} />
 */
export interface PaginationProps {
  current: number;
  total: number;
  onChange: (page: number) => void;
  className?: string;
}

export const Pagination: React.FC<PaginationProps> = ({
  current,
  total,
  onChange,
  className,
}) => {
  const pages = Array.from({ length: total }, (_, i) => i + 1);
  return (
    <nav className={cn("flex items-center gap-2", className)} aria-label="Pagination">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onChange(Math.max(1, current - 1))}
        disabled={current === 1}
      >
        Prev
      </Button>
      {pages.map((p) => (
        <Button
          key={p}
          variant={p === current ? "primary" : "ghost"}
          size="sm"
          onClick={() => onChange(p)}
        >
          {p}
        </Button>
      ))}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onChange(Math.min(total, current + 1))}
        disabled={current === total}
      >
        Next
      </Button>
    </nav>
  );
};
