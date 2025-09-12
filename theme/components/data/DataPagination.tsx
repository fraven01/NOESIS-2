/**
 * DataPagination renders previous/next controls.
 */
import * as React from 'react';

export interface DataPaginationProps {
  page: number;
  pageCount: number;
  onPageChange: (page: number) => void;
}

export function DataPagination({ page, pageCount, onPageChange }: DataPaginationProps) {
  return (
    <nav
      className="flex items-center justify-between p-2 border-t"
      aria-label="Pagination"
    >
      <button
        type="button"
        aria-label="Previous page"
        onClick={() => onPageChange(page - 1)}
        disabled={page <= 1}
        className="px-2 py-1 border rounded disabled:opacity-50"
      >
        Previous
      </button>
      <span className="text-sm">
        {page} / {pageCount}
      </span>
      <button
        type="button"
        aria-label="Next page"
        onClick={() => onPageChange(page + 1)}
        disabled={page >= pageCount}
        className="px-2 py-1 border rounded disabled:opacity-50"
      >
        Next
      </button>
    </nav>
  );
}
