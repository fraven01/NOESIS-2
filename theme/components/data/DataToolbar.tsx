/**
 * DataToolbar provides search, filter and bulk action controls for data tables.
 */
import * as React from 'react';

export interface DataToolbarProps {
  onSearch?: (value: string) => void;
  onFilter?: () => void;
  onBulkAction?: () => void;
  hasSelection?: boolean;
}

export function DataToolbar({
  onSearch,
  onFilter,
  onBulkAction,
  hasSelection = false,
}: DataToolbarProps) {
  return (
    <div
      role="toolbar"
      aria-label="Data toolbar"
      className="flex flex-wrap items-center gap-2 p-2 border-b"
    >
      <input
        aria-label="Search"
        type="search"
        className="px-2 py-1 border rounded"
        onChange={(e) => onSearch?.(e.target.value)}
      />
      <button
        type="button"
        aria-label="Filter"
        onClick={onFilter}
        className="px-2 py-1 border rounded"
      >
        Filter
      </button>
      <button
        type="button"
        aria-label="Bulk actions"
        onClick={onBulkAction}
        disabled={!hasSelection}
        className="px-2 py-1 border rounded disabled:opacity-50"
      >
        Bulk
      </button>
    </div>
  );
}
