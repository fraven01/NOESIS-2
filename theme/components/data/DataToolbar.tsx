import * as React from "react";
import * as Popover from "@radix-ui/react-popover";
import { Filter, Trash2 } from "lucide-react";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";

/**
 * Toolbar with search, filter popover and optional bulk actions.
 * @example
 * <DataToolbar search={search} onSearch={setSearch} selectedCount={selected.length} onDeleteSelected={clear} />
 */
export interface DataToolbarProps {
  search: string;
  onSearch: (value: string) => void;
  selectedCount: number;
  onDeleteSelected: () => void;
}

export const DataToolbar: React.FC<DataToolbarProps> = ({
  search,
  onSearch,
  selectedCount,
  onDeleteSelected,
  children,
}) => (
  <div className="flex flex-wrap items-center gap-2">
    <Input
      aria-label="Search"
      value={search}
      onChange={(e) => onSearch(e.target.value)}
      className="w-full max-w-xs"
    />
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button aria-label="Filter" variant="ghost" size="sm">
          <Filter className="h-4 w-4" />
        </Button>
      </Popover.Trigger>
      <Popover.Content className="rounded-xl border border-muted bg-bg p-4 shadow-sm" sideOffset={4}>
        {children || <div className="text-sm text-muted">No filters</div>}
      </Popover.Content>
    </Popover.Root>
    {selectedCount > 0 && (
      <Button
        aria-label="Delete selected"
        variant="destructive"
        size="sm"
        onClick={onDeleteSelected}
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    )}
  </div>
);
