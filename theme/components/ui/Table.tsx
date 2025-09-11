import * as React from "react";
import { cn } from "./cn";

/**
 * Table with optional sortable headers
 * @example
 * <Table>
 *  <TableHead>
 *    <TableRow>
 *      <SortableHeader id="name">Name</SortableHeader>
 *    </TableRow>
 *  </TableHead>
 * </Table>
 */
interface SortContextProps {
  sortKey: string | null;
  direction: "asc" | "desc";
  setSort: (key: string) => void;
}
const SortContext = React.createContext<SortContextProps | null>(null);

export interface TableProps extends React.TableHTMLAttributes<HTMLTableElement> {
  caption?: string;
}

export const Table: React.FC<TableProps> = ({ caption, className, children, ...props }) => {
  const [sortKey, setSortKey] = React.useState<string | null>(null);
  const [direction, setDirection] = React.useState<"asc" | "desc">("asc");
  const setSort = (key: string) => {
    if (sortKey === key) {
      setDirection(direction === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setDirection("asc");
    }
  };
  return (
    <SortContext.Provider value={{ sortKey, direction, setSort }}>
      <table className={cn("w-full text-left text-sm", className)} {...props}>
        {caption && <caption className="mb-2 text-left text-sm text-muted">{caption}</caption>}
        {children}
      </table>
    </SortContext.Provider>
  );
};

export const TableHead = (
  props: React.HTMLAttributes<HTMLTableSectionElement>
) => <thead {...props} />;
export const TableBody = (
  props: React.HTMLAttributes<HTMLTableSectionElement>
) => <tbody {...props} />;
export const TableRow = (
  props: React.HTMLAttributes<HTMLTableRowElement>
) => <tr {...props} />;
export const TableCell = (
  props: React.TdHTMLAttributes<HTMLTableCellElement>
) => <td className="px-4 py-2" {...props} />;

export interface SortableHeaderProps
  extends React.ThHTMLAttributes<HTMLTableCellElement> {
  id: string;
}

export const SortableHeader: React.FC<SortableHeaderProps> = ({
  id,
  className,
  children,
  ...props
}) => {
  const ctx = React.useContext(SortContext);
  const active = ctx?.sortKey === id;
  const ariaSort = active ? ctx?.direction : undefined;
  return (
    <th
      role="columnheader"
      aria-sort={ariaSort as any}
      onClick={() => ctx?.setSort(id)}
      className={cn("cursor-pointer select-none px-4 py-2", className)}
      {...props}
    >
      {children}
    </th>
  );
};
